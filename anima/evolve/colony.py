"""A colony: several creatures living in one arena, day after day — and,
when one of them clearly does better, a genetic algorithm.

Every creature has its own brain, body and thought log, and they all live in
one :class:`~anima.world.arena.SharedArena`, in rounds of 10 ms
(:mod:`~anima.evolve.hosting`).  They compete for food without having to
decide to, can share feasts that one alone cannot eat, hear each other and —
with a ``sense.kin`` port — sense where the others are.  Whether they keep
away from each other or stay together is up to what they learn.

At the end of each day (dawn, when they wake) the day is scored.  A
creature's **fitness** is the food it ate that day, in energy (a meal 0.25, a
share of a feast 0.4).  If the best creature **stands out** — it ate at least
``margin`` (25%) more than the median of the others, and at least
``min_gain`` (0.5, two meals) more — a generation turns over:

* the champion lives on, unchanged (elitism);
* every other place goes to a **variation of the champion**: its genes
  (:mod:`~anima.evolve.genome`) mutated, and with ``inherit="brain"`` (the
  default) a copy of its brain — synapses, memories — with a little synaptic
  noise, so the child starts from what the champion learned
  (exploit-and-explore, as in population-based training, Jaderberg et al.
  2017); with ``inherit="genes"`` a fresh brain built from the mutated genes,
  which must learn everything again (classic neuroevolution);
* ``crossover`` mixes the champion's genes with the runner-up's first;
* the creatures that lose their place are archived in ``retired/`` — or
  deleted with ``delete_old``.

If nobody stands out, everyone lives on into the next day as they are.
Children are born next to their parent.  (``selection="random"`` is a
control for experiments: the same turnover, but the parent is a creature
picked at random instead of the best.)

A colony is a directory: ``colony.json`` (settings, the arena, the lineage
of every creature that ever lived there, a report of every day), one
``<id>.npz`` brain file and ``<id>.thoughts.jsonl`` per living creature, and
``retired/``.  Any creature file can be loaded on its own
(``anima chat --brain colony/c3.npz``): it then lives alone in an ordinary
arena.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..body.body import BodyConfig
from ..body.motor import MotorConfig
from ..body.voice import VoiceConfig
from ..brain.builder import build_brain
from ..brain.presets import infant_spec
from ..runtime.agent import Agent
from ..runtime.persistence import load_state
from ..world.arena import SharedArena
from .genome import GENES, crossover, gene_diff, mutate, read_genes, write_genes
from .hosting import LocalHost, make_hosts

NAMES = ["Ada", "Bo", "Cyd", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jem", "Kit", "Lou", "Mae",
         "Ned", "Ola", "Pip", "Quin", "Rae", "Sam", "Tod", "Uma", "Vic", "Wyn", "Xan", "Yul", "Zed"]
NEAR = 3.0  # "with the others": another creature within this distance
TICK_S = 0.01


def creature_name(serial: int) -> str:
    base = NAMES[(serial - 1) % len(NAMES)]
    rnd = (serial - 1) // len(NAMES)
    return base if rnd == 0 else f"{base}{rnd + 1}"


@dataclass
class ColonyConfig:
    population: int = 4
    seed: int = 1
    cortex: int = 600  # cortical neurons per creature
    day_length_s: float = 1200.0
    margin: float = 0.25  # the champion must eat this much more than the others' median...
    min_gain: float = 0.5  # ...and at least this much more food energy (two meals)
    compare: str = "others"  # "others": best vs the others' median today; "itself": most improved since yesterday
    mutation: float = 0.2  # size of a gene mutation (log-normal sigma)
    mutation_rate: float = 0.5  # chance that each gene of a child mutates
    inherit: str = "brain"  # "brain": children start from a copy of the champion's brain; "genes": fresh
    crossover: bool = False  # mix the champion's genes with the runner-up's
    selection: str = "best"  # "best": the champion is the parent; "random": a random creature (a control)
    weight_noise: float = 0.05  # synaptic variation given to a "brain" child
    delete_old: bool = False  # delete creatures that lose their place instead of archiving them
    kin: bool = True  # creatures can sense where the others are
    thoughts: bool = True  # keep each creature's thought log
    arena: dict[str, Any] = field(default_factory=dict)  # SharedArena settings

    def check(self) -> None:
        if self.population < 2:
            raise ValueError("a colony needs at least 2 creatures")
        if self.inherit not in ("brain", "genes"):
            raise ValueError("inherit must be 'brain' or 'genes'")
        if self.compare not in ("others", "itself"):
            raise ValueError("compare must be 'others' or 'itself'")
        if self.selection not in ("best", "random"):
            raise ValueError("selection must be 'best' or 'random'")


#: settings that may be changed when a colony is resumed (the rest are fixed at birth)
POLICY_FIELDS = ("margin", "min_gain", "compare", "mutation", "mutation_rate", "inherit", "crossover",
                 "selection", "weight_noise", "delete_old", "thoughts")


def _body_config(cfg: dict[str, Any]) -> BodyConfig:
    cfg = dict(cfg)
    cfg["motor"] = MotorConfig(**cfg.get("motor", {}))
    cfg["voice"] = VoiceConfig(**cfg.get("voice", {}))
    if "hands" in cfg:
        cfg["hands"] = MotorConfig(**cfg["hands"])
    return BodyConfig(**cfg)


class DayStats:
    __slots__ = ("fitness", "meals", "feasts", "beat", "missed", "alone_at_feast", "utterances", "heard",
                 "near_s", "awake_s", "energy_sum", "energy_n", "energy_min")

    def __init__(self) -> None:
        self.fitness = 0.0
        self.meals = self.feasts = self.beat = self.missed = self.alone_at_feast = 0
        self.utterances = self.heard = 0
        self.near_s = self.awake_s = 0.0
        self.energy_sum, self.energy_n, self.energy_min = 0.0, 0, 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"fitness": round(self.fitness, 3), "meals": self.meals, "feasts": self.feasts,
                "beat_others": self.beat, "missed": self.missed, "alone_at_feast": self.alone_at_feast,
                "utterances": self.utterances, "heard_others": self.heard,
                "near_others": round(self.near_s / self.awake_s, 3) if self.awake_s else 0.0,
                "energy_mean": round(self.energy_sum / self.energy_n, 3) if self.energy_n else None,
                "energy_min": round(self.energy_min, 3)}


def decide_improved(fitness: dict[str, float], previous: dict[str, float], margin: float, min_gain: float
                    ) -> tuple[str | None, dict[str, Any]]:
    """Did one creature improve noticeably on its own previous day?  The one
    that improved most counts if it ate ``margin`` more than it did then, and
    at least ``min_gain`` more.  Creatures without a previous day cannot."""
    gains = sorted(((fitness[c] - previous[c], c) for c in fitness if c in previous), reverse=True)
    if not gains:
        return None, {"mode": "itself", "best": max(fitness, key=fitness.get), "stands_out": False,
                      "best_fitness": round(max(fitness.values()), 3), "previous": None, "gain": None,
                      "gain_rel": None, "needs": {"margin": margin, "min_gain": min_gain}}
    gain, cid = gains[0]
    f, p = fitness[cid], previous[cid]
    ok = gain >= min_gain and (p <= 0 or f >= (1.0 + margin) * p)
    tie = len(gains) > 1 and gains[1][0] == gain
    rel = gain / p if p > 0 else None
    info = {"mode": "itself", "best": cid, "best_fitness": round(f, 3), "previous": round(p, 3),
            "gain": round(gain, 3), "gain_rel": (round(rel, 3) if rel is not None else None),
            "needs": {"margin": margin, "min_gain": min_gain}, "stands_out": bool(ok and not tie)}
    return (cid if ok and not tie else None), info


def decide(fitness: dict[str, float], margin: float, min_gain: float) -> tuple[str | None, dict[str, Any]]:
    """Does one creature stand out?  Returns (its id or None, the numbers)."""
    ranked = sorted(fitness.items(), key=lambda kv: -kv[1])
    best_id, best = ranked[0]
    others = [f for _, f in ranked[1:]]
    median = float(np.median(others)) if others else 0.0
    gain = best - median
    rel = gain / median if median > 0 else (math.inf if gain > 0 else 0.0)
    ok = gain >= min_gain and (median <= 0 or best >= (1.0 + margin) * median)
    tie = len(ranked) > 1 and ranked[1][1] == best
    info = {"mode": "others", "best": best_id, "best_fitness": round(best, 3), "others_median": round(median, 3),
            "gain": round(gain, 3), "gain_rel": (round(rel, 3) if math.isfinite(rel) else None),
            "needs": {"margin": margin, "min_gain": min_gain}, "stands_out": bool(ok and not tie)}
    return (best_id if ok and not tie else None), info


class Colony:
    """See the module docstring.  ``workers`` > 1 runs the creatures in that
    many worker processes (same results, faster on a multi-core machine)."""

    def __init__(self, directory: str | Path, cfg: ColonyConfig, printer: Callable[[str], None] | None = print,
                 workers: int = 1, device: Any = None) -> None:
        cfg.check()
        self.dir = Path(directory)
        self.cfg = cfg
        self.printer = printer
        self.workers = max(1, int(workers))
        from ..core.device import resolve_device
        self.device = resolve_device(device)  # where the creatures' brains run (not part of the colony)
        self.rng = np.random.default_rng(cfg.seed * 7919 + 17)
        self.arena = SharedArena(**{"seed": cfg.seed * 31 + 5, **cfg.arena})
        self.hosts = make_hosts(self.workers, device=self.device)
        self.members: list[str] = []
        self.where: list[int] = []  # slot -> host
        self.names: list[str] = []
        self.energy: list[float] = []
        self.awake: list[bool] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.serial = 0
        self.day = 1
        self.t = 0.0
        self.phase0 = 0.3  # time of day (fraction) when the colony began
        self.history: list[dict[str, Any]] = []
        self.stats: dict[str, DayStats] = {}
        self._outbox: list[list[tuple[str, dict[str, Any]]]] = []
        self._heard: list[list[tuple[str, str, float]]] = []
        self._day_start_t = 0.0
        self._prev_phase: float | None = None
        self._ticks = 0
        self.speed_measured = 0.0
        self._listeners: list[tuple[str, Callable[[str, dict[str, Any]], None]]] = []
        self._closed = False

    # ================================================================ creation
    @classmethod
    def create(cls, directory: str | Path, cfg: ColonyConfig | None = None, founder: str | Path | None = None,
               printer: Callable[[str], None] | None = print, workers: int = 1, device: Any = None) -> "Colony":
        """A new colony: ``cfg.population`` young creatures with varied genes,
        or a copy of ``founder`` (an existing creature file, left untouched)
        plus variations of it."""
        cfg = cfg or ColonyConfig()
        col = cls(directory, cfg, printer, workers, device)
        try:
            col.dir.mkdir(parents=True, exist_ok=True)
            if founder is not None:
                state = load_state(founder)
                d = state["brain"]["drives"]
                cfg.day_length_s = float(d["cfg"]["day_length_s"])
                col.phase0 = (float(d["cfg"]["start_phase"]) + float(d["t_s"]) / cfg.day_length_s) % 1.0
                genes = read_genes(state)
                first = col._add(state, genes, parent=None, origin=f"copy of {Path(founder).name}", reseed=False)
                for _ in range(cfg.population - 1):
                    child, _ = mutate(genes, col.rng, cfg.mutation, cfg.mutation_rate)
                    col._add(state, child, parent=first, origin="variation of the founder",
                             jitter_sigma=cfg.weight_noise)
            else:
                wild = {g.name: g.default for g in GENES}
                for k in range(cfg.population):
                    genes = wild if k == 0 else mutate(wild, col.rng, cfg.mutation, cfg.mutation_rate)[0]
                    spec = infant_spec(cortex_size=cfg.cortex, seed=cfg.seed * 1000 + k + 1,
                                       day_length_s=cfg.day_length_s, kin=cfg.kin)
                    st = Agent(build_brain(spec, device="cpu"), None, BodyConfig()).state_dict()
                    col.phase0 = float(st["brain"]["drives"]["cfg"]["start_phase"])
                    col._add(st, genes, parent=None, origin="born in the colony")
            col._prev_phase = col.phase()
            col.save()
        except BaseException:
            col.close()
            raise
        return col

    def _add(self, state: dict[str, Any], genes: dict[str, float], parent: str | None, origin: str,
             slot: int | None = None, reseed: bool = True, jitter_sigma: float = 0.0,
             born_day: int | None = None, near: tuple[float, float] | None = None) -> str:
        """Make a creature from a state (with these genes) and give it a place."""
        self.serial += 1
        cid = f"c{self.serial}"
        name = creature_name(self.serial)
        state = write_genes(copy.deepcopy(state), genes)
        if slot is None:
            slot = self.arena.add(name)
            self.members.append(cid)
            self.where.append(slot % len(self.hosts))
            self.names.append(name)
            self.energy.append(float(state["body"]["energy"]))
            self.awake.append(True)
            self._outbox.append([])
            self._heard.append([])
        else:
            self.arena.reset(slot, name, near=near)
            self.members[slot], self.names[slot] = cid, name
            self.energy[slot] = float(state["body"]["energy"])
            self._outbox[slot], self._heard[slot] = [], []
        self.hosts[self.where[slot]].add(
            slot, state, name, {"type": "foraging", "kwargs": {"size": self.arena.size}},
            seed=int(self.rng.integers(1, 2 ** 31 - 1)) if reseed else None,
            jitter_sigma=jitter_sigma, jitter_seed=int(self.rng.integers(1, 2 ** 31 - 1)),
            thoughts_path=self._log_path(cid))
        gen = 0 if parent is None else int(self.records[parent]["generation"]) + 1
        self.records[cid] = {"id": cid, "name": name, "parent": parent, "generation": gen,
                             "born_day": born_day or self.day, "origin": origin,
                             "genes": {k: round(v, 5) for k, v in genes.items()}, "days": [], "fate": "alive"}
        self.stats[cid] = DayStats()
        return cid

    def _log_path(self, cid: str) -> str | None:
        return str(self.dir / f"{cid}.thoughts.jsonl") if self.cfg.thoughts else None

    # ================================================================ watching
    def listen(self, kind: str, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Call ``fn(creature_name, event)`` for every ``kind`` event of every
        member, present and future: ``thought``, ``utterance``, ``heard``,
        ``ate``, ``missed``, ``feast_alone``."""
        self._listeners.append((kind, fn))
        if kind == "thought":
            for h in self.hosts:
                h.set_forward(True)

    def _tell(self, kind: str, slot: int, data: dict[str, Any]) -> None:
        for k, fn in self._listeners:
            if k == kind:
                fn(self.names[slot], data)

    def agent(self, slot: int) -> Agent:
        """The living creature in ``slot`` (only when it lives in this process)."""
        host = self.hosts[self.where[slot]]
        if not isinstance(host, LocalHost):
            raise RuntimeError("the creatures live in worker processes (workers > 1)")
        return host.agents[slot]

    def info(self, slot: int) -> dict[str, Any]:
        return self.hosts[self.where[slot]].info(slot)

    # ================================================================ living
    def phase(self) -> float:
        """The creatures' time of day, as a fraction (they all share one clock)."""
        return (self.phase0 + self.t / self.cfg.day_length_s) % 1.0

    def clock(self) -> str:
        minutes = int(self.phase() * 24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @staticmethod
    def dawn() -> float:
        return 0.3  # the time of day new creatures are born at (DriveConfig.start_phase)

    def tick(self) -> None:
        """One round: 10 ms for everyone."""
        a, n = self.arena, len(self.members)
        inputs: list[dict[int, dict[str, Any]]] = [{} for _ in self.hosts]
        for i in range(n):
            inputs[self.where[i]][i] = {"obs": a.observe(i), "smell": a.smell_ahead(i),
                                        "events": self._outbox[i], "heard": self._heard[i]}
            self._outbox[i], self._heard[i] = [], []
        for h, inp in zip(self.hosts, inputs):
            if inp:
                h.step_send(inp)
        out: dict[int, dict[str, Any]] = {}
        for h, inp in zip(self.hosts, inputs):
            if inp:
                out.update(h.step_recv())
        # everyone's movements, in a shuffled order (nobody is always first)
        for i in self.rng.permutation(n):
            for action, before in out[i]["actions"]:
                evs = a.apply(i, action)
                after = a.smell_ahead(i)
                self._outcomes(i, evs)
                self._outbox[i].append(("approach", {"rel": (after - before) / (0.5 * (after + before) + 0.02)}))
        for j in range(n):
            pending = a.take_pending(j)
            if pending:
                self._outcomes(j, pending)
        for i in range(n):
            o = out[i]
            self.energy[i], self.awake[i] = o["energy"], o["awake"]
            for text in o["said"]:
                self._spoke(i, text)
            for rec in o["thoughts"]:
                self._tell("thought", i, {**rec, "type": rec["kind"], "kind": "thought"})
            a.step_body(i, TICK_S)
        a.step(TICK_S)
        self.t += TICK_S
        self._ticks += 1
        if self._ticks % 10 == 0:
            self._sample(10 * TICK_S)
        phase, dawn = self.phase(), self.dawn()
        prev = self._prev_phase if self._prev_phase is not None else phase
        crossed = (prev < dawn <= phase) or (prev > phase and (dawn > prev or dawn <= phase))
        self._prev_phase = phase
        if crossed and self.t - self._day_start_t > 0.5 * self.cfg.day_length_s:
            self.end_of_day()

    def _outcomes(self, i: int, evs: list[tuple[str, dict[str, Any]]]) -> None:
        """What happened to creature ``i`` in the arena: keep score, pass it on next round."""
        st = self.stats[self.members[i]]
        for kind, data in evs:
            self._outbox[i].append((kind, data))
            if kind == "ate":
                st.fitness += float(data.get("energy", 0.25))
                if data.get("feast"):
                    st.feasts += 1
                else:
                    st.meals += 1
                    st.beat += bool(data.get("beat"))
            elif kind == "missed":
                st.missed += 1
            elif kind == "feast_alone":
                st.alone_at_feast += 1
            if kind in ("ate", "missed", "feast_alone"):
                self._tell(kind, i, dict(data))

    def _spoke(self, i: int, text: str) -> None:
        """Creature ``i`` said something: the others nearby hear it next round."""
        self.stats[self.members[i]].utterances += 1
        self._tell("utterance", i, {"text": text})
        r = self.arena.hear_radius
        for j in range(len(self.members)):
            if j != i:
                d = self.arena.distance(i, j)
                if d <= r:
                    loud = max(0.2, 1.0 - d / r)
                    self._heard[j].append((text, self.names[i], loud))
                    self.stats[self.members[j]].heard += 1
                    self._tell("heard", j, {"text": text, "from": self.names[i], "loudness": loud})

    def _sample(self, dt: float) -> None:
        n = len(self.members)
        for i in range(n):
            st = self.stats[self.members[i]]
            if self.awake[i]:
                st.awake_s += dt
                if any(self.arena.distance(i, j) <= NEAR for j in range(n) if j != i):
                    st.near_s += dt
            st.energy_sum += self.energy[i]
            st.energy_n += 1
            st.energy_min = min(st.energy_min, self.energy[i])

    def run(self, days: int | None = None, seconds: float | None = None, realtime: bool = False,
            speed: float = 1.0, stop: Callable[[], bool] | None = None, report_every_h: float = 3.0) -> None:
        """Live until ``days`` more days have ended (or for ``seconds``), as
        fast as the CPU allows or paced to the wall clock (x ``speed``).
        ``stop()`` is checked every round."""
        until_day = self.day + int(days) if days is not None else None
        end = self.t + seconds if seconds is not None else math.inf
        if until_day is None and seconds is None:
            until_day = self.day + 1
        sim0, wall0 = self.t, time.perf_counter()
        bucket = self._bucket(report_every_h)
        while self.t < end - 1e-9 and (until_day is None or self.day < until_day) \
                and not (stop is not None and stop()):
            self.tick()
            b = self._bucket(report_every_h)
            if b != bucket:
                bucket = b
                if b is not None:
                    self._say(self.summary_line())
            if self._ticks % 50 == 0:
                now = time.perf_counter()
                self.speed_measured = (self.t - sim0) / max(1e-9, now - wall0)
                if realtime:
                    delay = wall0 + (self.t - sim0) / speed - now
                    if delay > 0:
                        time.sleep(delay)

    def _bucket(self, every_h: float) -> int | None:
        """Which ``every_h``-hour stretch of the creatures' day it is (for progress lines)."""
        return None if not every_h else int(self.phase() * 24.0 // every_h)

    # ================================================================ the end of a day
    def end_of_day(self) -> dict[str, Any]:
        cfg = self.cfg
        rows = []
        for cid in self.members:
            st = self.stats[cid].as_dict()
            rec = self.records[cid]
            rec["days"].append({"day": self.day, **st})
            rows.append({"id": cid, "name": rec["name"], "generation": rec["generation"], **st})
        fitness = {r["id"]: r["fitness"] for r in rows}
        if cfg.compare == "itself":
            previous = {c: self.records[c]["days"][-2]["fitness"] for c in self.members
                        if len(self.records[c]["days"]) >= 2}
            champion, info = decide_improved(fitness, previous, cfg.margin, cfg.min_gain)
        else:
            champion, info = decide(fitness, cfg.margin, cfg.min_gain)
        if champion and cfg.selection == "random":  # the control: same turnover, a parent picked blindly
            champion = str(self.rng.choice(sorted(self.members)))
            info["parent"] = champion
        report: dict[str, Any] = {"day": self.day, "t": round(self.t, 2), "clock": self.clock(),
                                  "creatures": rows, "arena": dict(self.arena.counts), "decision": info}
        report["next"] = self._next_generation(champion, fitness) if champion else {"kept": list(self.members)}
        self.history.append(report)
        self._say(self.day_report(report))
        self.day += 1
        self._day_start_t = self.t
        for cid in self.members:
            self.stats[cid] = DayStats()
        self.arena.reset_counts()
        self.save()
        return report

    def _next_generation(self, champion: str, fitness: dict[str, float]) -> dict[str, Any]:
        cfg = self.cfg
        c_slot = self.members.index(champion)
        parent_state = self.hosts[self.where[c_slot]].state(c_slot)
        home = (self.arena.bodies[c_slot]["x"], self.arena.bodies[c_slot]["y"])
        parent_genes = dict(self.records[champion]["genes"])
        others = sorted((f, cid) for cid, f in fitness.items() if cid != champion)
        runner_genes = dict(self.records[others[-1][1]]["genes"]) if others else None
        retired, born = [], []
        for slot, cid in list(enumerate(self.members)):
            if cid != champion:
                self._retire(slot, cid)
                retired.append(cid)
        for slot in range(len(self.members)):
            if slot == c_slot:
                continue
            cross = cfg.crossover and runner_genes is not None
            base = crossover(parent_genes, runner_genes, self.rng) if cross else parent_genes
            genes, _ = mutate(base, self.rng, cfg.mutation, cfg.mutation_rate)
            if cfg.inherit == "brain":
                state, sigma, origin = parent_state, cfg.weight_noise, "copy of its parent's brain"
            else:
                state, sigma, origin = self._fresh_like(parent_state), 0.0, "new brain"
            cid = self._add(state, genes, parent=champion, slot=slot, jitter_sigma=sigma, born_day=self.day + 1,
                            origin=origin + (", genes crossed with the runner-up" if cross else ""), near=home)
            self.records[cid]["mutations"] = gene_diff(parent_genes, genes)
            born.append(cid)
        return {"champion": champion, "born": born, "retired": retired,
                "retired_files": "deleted" if cfg.delete_old else "archived in retired/"}

    def _fresh_like(self, parent_state: dict[str, Any]) -> dict[str, Any]:
        """A fresh brain of the parent's design (new wiring), born at the colony's time of day."""
        spec = copy.deepcopy(parent_state["spec"])
        spec["seed"] = int(self.rng.integers(1, 2 ** 31 - 1))
        st = Agent(build_brain(spec, device="cpu"), None, _body_config(parent_state["body"]["cfg"])).state_dict()
        st["brain"]["drives"]["t_s"] = parent_state["brain"]["drives"]["t_s"]
        return st

    def _retire(self, slot: int, cid: str) -> None:
        """A creature leaves the colony: archived (its final state saved in
        ``retired/``), or deleted with ``delete_old``."""
        rec = self.records[cid]
        rec["fate"] = "retired"
        rec["retired_day"] = self.day
        host = self.hosts[self.where[slot]]
        log = self.dir / f"{cid}.thoughts.jsonl"
        logs = [log, log.with_name(log.name + ".1")]
        if self.cfg.delete_old:
            host.remove(slot)
            for p in [self.dir / f"{cid}.npz", *logs]:
                p.unlink(missing_ok=True)
            rec["files"] = "deleted"
        else:
            arch = self.dir / "retired"
            arch.mkdir(exist_ok=True)
            host.remove(slot, save_to=str(arch / f"{cid}.npz"), snapshot=self.arena.snapshot(slot))
            (self.dir / f"{cid}.npz").unlink(missing_ok=True)
            for p in logs:
                if p.exists():
                    p.replace(arch / p.name)
            rec["files"] = f"retired/{cid}.npz"
        self.stats.pop(cid, None)

    # ================================================================ text
    def _say(self, text: str) -> None:
        if self.printer is not None:
            self.printer(text)

    def summary_line(self) -> str:
        parts = []
        for i, cid in enumerate(self.members):
            st = self.stats[cid]
            z = "" if self.awake[i] else " (asleep)"
            parts.append(f"{self.names[i]} {st.meals}+{st.feasts}f e{self.energy[i]:.2f}{z}")
        c = self.arena.counts
        return (f"[day {self.day} {self.clock()}] " + " | ".join(parts)
                + f" | feasts {c['feasts']}, contested meals {c['contested']}")

    def day_report(self, report: dict[str, Any]) -> str:
        lines = [f"=== end of day {report['day']} ===",
                 "  creature        gen  fitness  meals  feasts  beat  missed  near others  heard  mean energy"]
        for r in sorted(report["creatures"], key=lambda r: -r["fitness"]):
            em = "-" if r["energy_mean"] is None else f"{r['energy_mean']:.2f}"
            who = f"{r['name']} ({r['id']})"
            lines.append(f"  {who:<15} {r['generation']:>3}  {r['fitness']:7.2f}  {r['meals']:5d}"
                         f"  {r['feasts']:6d}  {r['beat_others']:4d}  {r['missed']:6d}  {r['near_others']:10.0%}"
                         f"  {r['heard_others']:5d}  {em:>11}")
        d = report["decision"]
        best = self.records[d["best"]]["name"]
        rel = "" if d["gain_rel"] is None else f" ({d['gain_rel']:+.0%})"
        itself = d.get("mode") == "itself"
        if itself:
            was = "no previous day" if d["previous"] is None else f"{d['previous']:.2f} the day before"
            measure = f"{d['best_fitness']:.2f} vs {was}{rel}"
        else:
            measure = f"{d['best_fitness']:.2f} vs the others' median {d['others_median']:.2f}{rel}"
        if d["stands_out"]:
            nxt = report["next"]
            kids = ", ".join(f"{self.records[c]['name']} ({self.records[c].get('mutations', '')})"
                             for c in nxt["born"])
            gone = ", ".join(self.records[c]["name"] for c in nxt["retired"])
            parent = self.records[nxt["champion"]]["name"]
            lines.append(f"  -> {best} {'improved most' if itself else 'stands out'}: {measure}.")
            if d.get("parent"):
                lines.append(f"     (selection='random', a control: the parent is picked at random - {parent})")
            lines.append(f"     Next day: {parent} and its variations: {kids}.")
            lines.append(f"     Retired: {gone} ({nxt['retired_files']}).")
        else:
            lines.append(f"  -> nobody stands out ({'most improved' if itself else 'best'} {best} {measure}; "
                         f"needs +{d['needs']['margin']:.0%} and +{d['needs']['min_gain']:g}): everyone lives on.")
        return "\n".join(lines)

    def lineage(self) -> str:
        """Every creature that lived here, as a family tree."""
        kids: dict[str | None, list[str]] = {}
        for cid, r in self.records.items():
            kids.setdefault(r["parent"], []).append(cid)
        lines: list[str] = []

        def walk(cid: str, depth: int) -> None:
            r = self.records[cid]
            fit = ", ".join(f"{d['fitness']:.2f}" for d in r["days"]) or "-"
            fate = "alive" if r["fate"] == "alive" else f"retired after day {r.get('retired_day')}"
            lines.append(f"{'  ' * depth}{r['name']} ({cid}, gen {r['generation']}, born day {r['born_day']}, "
                         f"{fate}) food by day: {fit}")
            for k in sorted(kids.get(cid, []), key=lambda c: int(c[1:])):
                walk(k, depth + 1)

        for root in sorted(kids.get(None, []), key=lambda c: int(c[1:])):
            walk(root, 0)
        return "\n".join(lines)

    # ================================================================ persistence
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for slot, cid in enumerate(self.members):
            self.hosts[self.where[slot]].save(slot, str(self.dir / f"{cid}.npz"), self.arena.snapshot(slot))
        data = {"version": 1, "config": asdict(self.cfg), "day": self.day, "t": self.t, "serial": self.serial,
                "phase0": self.phase0, "members": list(self.members), "records": self.records,
                "history": self.history, "arena": self.arena.state_dict(), "rng": self.rng.bit_generator.state,
                "stats": {cid: {k: getattr(s, k) for k in DayStats.__slots__} for cid, s in self.stats.items()},
                "energy": self.energy, "awake": self.awake, "outbox": self._outbox, "heard": self._heard,
                "prev_phase": self._prev_phase, "day_start_t": self._day_start_t, "ticks": self._ticks}
        tmp = self.dir / "colony.json.tmp"
        tmp.write_text(json.dumps(data, default=_jsonable), encoding="utf-8")
        tmp.replace(self.dir / "colony.json")

    @classmethod
    def load(cls, directory: str | Path, printer: Callable[[str], None] | None = print, workers: int = 1,
             device: Any = None, **policy: Any) -> "Colony":
        d = Path(directory)
        data = json.loads((d / "colony.json").read_text(encoding="utf-8"))
        cfg = ColonyConfig(**data["config"])
        for k, v in policy.items():
            if k not in POLICY_FIELDS:
                raise ValueError(f"{k} is fixed when a colony is born")
            if v is not None:
                setattr(cfg, k, v)
        col = cls(d, cfg, printer, workers, device)
        try:
            col.arena = SharedArena.from_state(data["arena"])
            col.day, col.t, col.serial = int(data["day"]), float(data["t"]), int(data["serial"])
            col.phase0 = float(data.get("phase0", 0.3))
            col.records, col.history = data["records"], data["history"]
            col.rng.bit_generator.state = data["rng"]
            col._ticks = int(data.get("ticks", 0))
            for slot, cid in enumerate(data["members"]):
                col.members.append(cid)
                col.where.append(slot % len(col.hosts))
                col.names.append(col.records[cid]["name"])
                col.hosts[col.where[slot]].load(slot, str(d / f"{cid}.npz"), col._log_path(cid))
                st = DayStats()
                for k, v in data.get("stats", {}).get(cid, {}).items():
                    setattr(st, k, v)
                col.stats[cid] = st
            n = len(col.members)
            col.energy = [float(x) for x in data.get("energy", [0.75] * n)]
            col.awake = [bool(x) for x in data.get("awake", [True] * n)]
            col._outbox = [[(k, dict(v)) for k, v in box] for box in data.get("outbox", [[] for _ in range(n)])]
            col._heard = [[(t, s, float(lv)) for t, s, lv in box]
                          for box in data.get("heard", [[] for _ in range(n)])]
            col._prev_phase = data.get("prev_phase")
            col._day_start_t = float(data.get("day_start_t", 0.0))
        except BaseException:
            col.close()
            raise
        return col

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for h in self.hosts:
            h.close()

    def __enter__(self) -> "Colony":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return str(x)


__all__ = ["Colony", "ColonyConfig", "DayStats", "decide", "decide_improved", "creature_name", "NAMES",
           "POLICY_FIELDS"]
