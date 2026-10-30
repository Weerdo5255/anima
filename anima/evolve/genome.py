"""What a creature passes on: its genes.

A gene here is one inborn setting of a creature — how noisy (exploratory)
its movements are, how fast its sensorimotor synapses learn, how long a
movement stays eligible for credit, how rewarding company or novelty is to
it, how readily it moves.  Genes are *not* its synapses: those it learns
while it lives.  (A child can still be given a copy of its parent's
synapses — see ``inherit`` in :class:`~anima.evolve.colony.ColonyConfig`.)

Genes are read from and written into a creature's saved state (its brain
spec, its brain settings and its body settings), so a creature with new
genes is built, saved and loaded like any other.

Mutation is multiplicative (log-normal): a gene is multiplied by
``exp(sigma * z)``, ``z`` standard normal, and kept within its bounds — so
"a bit more" and "a bit less" are equally likely and no gene can flip sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MOVE_RULES = ("sense.bearing->motor", "sense.wall->motor", "cortex.E->motor", "sense.kin->motor")
VOICE_RULES = ("cortex.E->vocal.gate", "intero->vocal.C", "intero->vocal.V", "cortex.E->vocal.C",
               "cortex.E->vocal.V")


@dataclass(frozen=True)
class Gene:
    name: str
    about: str
    where: tuple[Any, ...]  # ("pop", name, field) | ("rule", names, field) | ("effects"|"drives"|"body"|"motor", field)
    default: float
    low: float
    high: float


GENES: tuple[Gene, ...] = (
    Gene("explore", "noise in its movement channels (how much it tries things)", ("pop", "motor", "noise"),
         3.5, 1.5, 6.0),
    Gene("eagerness", "how little activity it takes to start a movement (threshold; lower = moves sooner)",
         ("motor", "threshold"), 0.28, 0.15, 0.45),
    Gene("move_lr", "learning rate of the synapses that steer it", ("rule", MOVE_RULES, "lr"), 0.05, 0.01, 0.25),
    Gene("move_trace", "how long (ms) a movement stays eligible for credit", ("rule", MOVE_RULES, "tau_e"),
         300.0, 100.0, 1500.0),
    Gene("appetite", "reward for getting closer to the smell of food", ("body", "approach_reward"),
         1.0, 0.2, 4.0),
    Gene("vigor", "how much a good run of rewards energises its movements", ("effects", "vigor"),
         3.0, 0.5, 8.0),
    Gene("curiosity", "reward for novelty when bored", ("effects", "curiosity_reward"), 0.6, 0.05, 2.5),
    Gene("sociability", "reward for company (per unit of loneliness relieved)", ("effects", "social_reward"),
         1.5, 0.2, 5.0),
    Gene("loneliness", "how slowly it gets lonely (seconds)", ("drives", "loneliness_rise_s"), 90.0, 20.0, 400.0),
    Gene("chatty", "noise in its urge to speak", ("pop", "vocal.gate", "noise"), 3.0, 1.5, 5.0),
    Gene("voice_lr", "learning rate of what it says", ("rule", VOICE_RULES, "lr"), 0.06, 0.01, 0.25),
)
GENE_NAMES = tuple(g.name for g in GENES)


def _proj_name(q: dict[str, Any]) -> str:
    return q.get("name") or f"{q['pre']}->{q['post']}"


def _slot(state: dict[str, Any], gene: Gene) -> list[tuple[dict[str, Any], str]]:
    """(dict, key) pairs where the gene lives in a creature state."""
    kind = gene.where[0]
    if kind == "pop":
        _, pop, field = gene.where
        return [(p, field) for p in state["spec"]["population"] if p["name"] == pop]
    if kind == "rule":
        _, names, field = gene.where
        return [(q["rule"], field) for q in state["spec"]["projection"]
                if _proj_name(q) in names and isinstance(q.get("rule"), dict)]
    if kind == "effects":
        return [(state["brain"]["effects"], gene.where[1])]
    if kind == "drives":
        return [(state["brain"]["drives"]["cfg"], gene.where[1])]
    if kind == "body":
        return [(state["body"]["cfg"], gene.where[1])]
    if kind == "motor":
        return [(state["body"]["cfg"]["motor"], gene.where[1])]
    raise ValueError(f"unknown gene location {gene.where}")


def read_genes(state: dict[str, Any]) -> dict[str, float]:
    """The genes of a creature, from its state (``agent.state_dict()``)."""
    out = {}
    for g in GENES:
        slots = _slot(state, g)
        out[g.name] = float(slots[0][0].get(slots[0][1], g.default)) if slots else g.default
    return out


def write_genes(state: dict[str, Any], genes: dict[str, float]) -> dict[str, Any]:
    """Write genes into a creature state (in place; also returned).  Genes
    its brain has no place for (e.g. ``sense.kin`` rules) are skipped."""
    for g in GENES:
        if g.name in genes:
            for d, key in _slot(state, g):
                d[key] = float(np.clip(genes[g.name], g.low, g.high))
    return state


def mutate(genes: dict[str, float], rng: np.random.Generator, sigma: float = 0.2, rate: float = 0.5
           ) -> tuple[dict[str, float], list[str]]:
    """A varied copy of ``genes``: each gene changes with probability
    ``rate`` (at least one always does).  Returns (genes, names changed)."""
    out = dict(genes)
    hit = [g for g in GENES if rng.random() < rate]
    if not hit:
        hit = [GENES[int(rng.integers(len(GENES)))]]
    for g in hit:
        x = out.get(g.name, g.default)
        out[g.name] = float(np.clip(x * math.exp(sigma * rng.standard_normal()), g.low, g.high))
    return out, [g.name for g in hit]


def crossover(a: dict[str, float], b: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    """Uniform crossover: each gene from ``a`` or ``b`` with equal chance."""
    return {g.name: (a if rng.random() < 0.5 else b).get(g.name, g.default) for g in GENES}


def gene_diff(parent: dict[str, float], child: dict[str, float]) -> str:
    """Changes as text, e.g. ``explore 3.50->4.10 (+17%)``."""
    parts = []
    for g in GENES:
        p, c = parent.get(g.name, g.default), child.get(g.name, g.default)
        if abs(c - p) > 1e-9 * max(1.0, abs(p)):
            parts.append(f"{g.name} {p:.3g}->{c:.3g} ({(c / p - 1) * 100:+.0f}%)")
    return ", ".join(parts) or "no change"


__all__ = ["Gene", "GENES", "GENE_NAMES", "read_genes", "write_genes", "mutate", "crossover", "gene_diff"]
