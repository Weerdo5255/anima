"""Experiment: does the creature learn to forage?

The infant brain starts with no useful wiring between "where the food smell
comes from" and "which way to move": those synapses are weak and random, so
its first movements are random.  Each movement charges eligibility traces;
dopamine — the critic's reward prediction error, driven by the food smell
getting stronger and by eating — turns them into lasting weight changes.

Measurement (a behavioural probe, plasticity paused while probing):
the food is held at a fixed bearing — 90 deg left, straight ahead, 90 deg
right — and we count what the creature chooses over 40 decisions.  The
correct choices are left / forward / right; chance is 1/3.

As in animal learning studies the creature is kept mildly food-restricted
(energy capped at ``energy_cap``) so food stays motivating, and probes are
run at a fixed hunger.  Control: the same brain with plasticity frozen.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..runtime.agent import Agent

PROBE_BEARINGS = (90, 0, -90)
CORRECT = {90: "left", 0: "forward", -90: "right"}


def probe_policy(agent: Agent, bearing_deg: float, n_actions: int = 40, distance: float = 5.0,
                 energy: float = 0.4) -> dict[str, int]:
    """Choice counts with the food held at ``bearing_deg`` (plasticity paused).

    Runs on the agent's own clock (it consumes simulated time) but restores
    the world and the body's energy afterwards.
    """
    net = agent.brain.net
    if not agent.brain.awake:
        raise RuntimeError("probe_policy: the creature is asleep (use a longer day_length_s)")
    was_plastic = net.plasticity_enabled
    net.plasticity_enabled = False
    w = agent.world
    saved = w.state_dict()
    saved_energy = agent.body.energy
    got: list[str] = []
    listener = lambda e: got.append(e["action"])  # noqa: E731
    agent.on("action", listener)

    def place() -> None:
        w.x, w.y, w.heading = w.size / 2, w.size / 2, 0.0
        th = math.radians(bearing_deg)
        w.food = [[w.x + distance * math.cos(th), w.y + distance * math.sin(th)]]
        w.regrow = []
        agent.body.energy = energy

    place()
    limit = agent.t_ms + 120_000
    while len(got) < n_actions and agent.t_ms < limit:
        agent.tick()
        place()
    agent._listeners["action"].remove(listener)
    w.load_state_dict(saved)
    agent.body.energy = saved_energy
    net.plasticity_enabled = was_plastic
    counts = {"forward": 0, "left": 0, "right": 0}
    for g in got:
        counts[g] += 1
    return counts


def correct_rate(probes: dict[int, dict[str, int]]) -> float:
    rates = []
    for b, counts in probes.items():
        n = sum(counts.values())
        if n:
            rates.append(counts[CORRECT[b]] / n)
    return float(np.mean(rates)) if rates else float("nan")


def run_foraging(seconds: float = 600.0, seed: int = 1, plastic: bool = True,
                 probe_every: float = 150.0, energy_cap: float = 0.5, n_actions: int = 40,
                 day_length_s: float = 7200.0, verbose: bool = False,
                 **create_kwargs: Any) -> dict[str, Any]:
    """Live ``seconds`` of foraging, probing the policy every ``probe_every``.

    The day is made long (2 h by default) so the creature stays awake for the
    whole experiment; sleep is studied separately.
    """
    agent = Agent.create(seed=seed, day_length_s=day_length_s, **create_kwargs)
    agent.brain.net.plasticity_enabled = plastic
    eaten: list[float] = []
    agent.on("ate", lambda e: eaten.append(e["t"]))
    lived = 0.0
    points = []

    def probe() -> None:
        probes = {b: probe_policy(agent, b, n_actions=n_actions) for b in PROBE_BEARINGS}
        rate = correct_rate(probes)
        points.append({"t": lived, "correct": rate, "probes": probes,
                       "food_so_far": len(eaten)})
        if verbose:
            tag = "plastic" if plastic else "frozen "
            print(f"  [{tag} seed {seed}] after {lived:5.0f}s of life: correct {rate:.2f}  "
                  f"(L {probes[90]['left']}/{n_actions}, F {probes[0]['forward']}/{n_actions}, "
                  f"R {probes[-90]['right']}/{n_actions})  food eaten {len(eaten)}", flush=True)

    probe()
    ticks_per_chunk = int(probe_every * 1000 / agent.tick_ms)
    while lived < seconds - 1e-6:
        for _ in range(ticks_per_chunk):
            agent.tick()
            if agent.body.energy > energy_cap:
                agent.body.energy = energy_cap
        lived += probe_every
        probe()
    return {"seed": seed, "plastic": plastic, "points": points, "eaten_t": eaten,
            "seconds": seconds}


def main(seconds: float = 600.0, seeds: tuple[int, ...] = (1, 2), plot: str | None = None,
         verbose: bool = True, probe_every: float = 150.0) -> dict[str, Any]:
    results: dict[str, Any] = {"plastic": [], "frozen": []}
    for cond in ("plastic", "frozen"):
        for s in seeds:
            results[cond].append(run_foraging(seconds, seed=s, plastic=(cond == "plastic"),
                                              probe_every=probe_every, verbose=verbose))
    if verbose:
        for cond in ("plastic", "frozen"):
            first = np.mean([r["points"][0]["correct"] for r in results[cond]])
            last = np.mean([r["points"][-1]["correct"] for r in results[cond]])
            print(f"{cond}: correct choices {first:.2f} -> {last:.2f} (chance 0.33)")
    if plot:
        _plot(results, plot)
    return results


def _plot(results: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for cond, color in (("plastic", "#1f9d6b"), ("frozen", "#9a9a9a")):
        runs = results[cond]
        for key, ax in (("correct", axes[0]), ("food_so_far", axes[1])):
            for i, r in enumerate(runs):
                x = [p["t"] / 60.0 for p in r["points"]]
                y = [p[key] for p in r["points"]]
                ax.plot(x, y, "-o", color=color, ms=3.5, lw=1.4, alpha=0.9,
                        label=cond if i == 0 else None)
    axes[0].axhline(1 / 3, color="k", lw=0.7, ls=":")
    axes[0].text(0.05, 0.345, "chance", fontsize=8)
    axes[0].set(title="correct move with food left / ahead / right", xlabel="minutes of life",
                ylabel="fraction correct", ylim=(0, 1))
    axes[1].set(title="food found (cumulative)", xlabel="minutes of life")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
