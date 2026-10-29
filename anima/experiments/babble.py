"""Experiment: can you shape what it babbles by smiling at it?

Human infants' babbling is shaped by contingent social feedback (Goldstein,
King & West 2003).  Here a scripted caregiver smiles (non-verbal ``:)`` —
no sound, so the auditory system plays no part) about half a second after
every syllable that starts with a target consonant, and ignores all other
babbling.  The smile relieves loneliness and adds approval: dopamine.
Reward-gated plasticity in the vocal system should make the target
consonant more common.

Controls:
* ``frozen``  same caregiver, plasticity off;
* ``yoked``   same number of smiles, but delivered at random moments,
              not contingent on the target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..runtime.agent import Agent


class Caregiver:
    """Responds to target syllables (or at random, if yoked)."""

    def __init__(self, agent: Agent, target: str, mode: str = "contingent", delay_s: float = 0.5,
                 refractory_s: float = 2.0, rate_hint: float = 0.1, seed: int = 0,
                 start_s: float = 0.0) -> None:
        self.agent = agent
        self.target = target
        self.mode = mode
        self.delay_s = delay_s
        self.refractory_s = refractory_s
        self.rate_hint = rate_hint
        self.start_s = start_s
        self.rng = np.random.default_rng(seed)
        self.pending: list[float] = []
        self.last = -1e9
        self.responses = 0
        agent.on("syllable", self._on_syllable)

    def _on_syllable(self, e: dict[str, Any]) -> None:
        t = e["t"]
        if t < self.start_s or self.mode != "contingent":
            return
        if e["text"].startswith(self.target) and t - self.last > self.refractory_s:
            self.pending.append(t + self.delay_s)
            self.last = t

    def tick(self) -> None:
        t = self.agent.t_ms / 1000.0
        if t < self.start_s:
            return
        if self.mode == "yoked" and self.rng.random() < self.rate_hint * self.agent.tick_ms / 1000.0:
            self.pending.append(t)
        while self.pending and self.pending[0] <= t:
            self.pending.pop(0)
            self.agent.hear(":)")
            self.responses += 1


def run_babble(seconds: float = 600.0, seed: int = 1, target: str = "m", mode: str = "contingent",
               plastic: bool = True, baseline_s: float = 120.0, yoked_rate: float = 0.1,
               verbose: bool = False) -> dict[str, Any]:
    agent = Agent.create(seed=seed, world="void")
    agent.brain.net.plasticity_enabled = plastic
    syl: list[tuple[float, str]] = []
    agent.on("syllable", lambda e: syl.append((e["t"], e["text"])))
    care = Caregiver(agent, target, mode=mode, seed=seed + 100, start_s=baseline_s,
                     rate_hint=yoked_rate)
    n_ticks = int(seconds * 1000 / agent.tick_ms)
    for k in range(n_ticks):
        agent.body.energy = 0.8  # well fed: we study the voice here
        care.tick()
        agent.tick()
        if verbose and k % 6000 == 0 and k:
            share = target_share(syl, target, agent.t_ms / 1000 - 60, agent.t_ms / 1000)
            print(f"  [{mode}{'' if plastic else '/frozen'} seed {seed}] t={agent.t_ms / 1000:5.0f}s "
                  f"'{target}' share {share:.2f}  syllables {len(syl)}  smiles {care.responses}",
                  flush=True)
    return {"syllables": syl, "responses": care.responses, "target": target}


def target_share(syl: list[tuple[float, str]], target: str, t0: float, t1: float) -> float:
    sel = [s for t, s in syl if t0 <= t < t1]
    if not sel:
        return float("nan")
    return sum(s.startswith(target) for s in sel) / len(sel)


def curve(res: dict[str, Any], seconds: float, bins: int = 5) -> list[dict[str, float]]:
    edges = np.linspace(0, seconds, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = sum(1 for t, _ in res["syllables"] if a <= t < b)
        out.append({"t0": float(a), "t1": float(b), "share": target_share(res["syllables"], res["target"], a, b),
                    "syllables_per_min": n / ((b - a) / 60.0)})
    return out


def main(seconds: float = 600.0, seeds: tuple[int, ...] = (1, 2), target: str = "m",
         plot: str | None = None, verbose: bool = True, baseline_s: float = 120.0) -> dict[str, Any]:
    results: dict[str, Any] = {"contingent": [], "yoked": [], "frozen": []}
    for s in seeds:
        r = run_babble(seconds, seed=s, target=target, mode="contingent", plastic=True,
                       baseline_s=baseline_s, verbose=verbose)
        results["contingent"].append({"curve": curve(r, seconds), "responses": r["responses"]})
        # yoked control: the same number of smiles, delivered at random moments
        rate = r["responses"] / max(1.0, seconds - baseline_s)
        y = run_babble(seconds, seed=s, target=target, mode="yoked", plastic=True,
                       baseline_s=baseline_s, yoked_rate=rate, verbose=verbose)
        results["yoked"].append({"curve": curve(y, seconds), "responses": y["responses"]})
        f = run_babble(seconds, seed=s, target=target, mode="contingent", plastic=False,
                       baseline_s=baseline_s, verbose=verbose)
        results["frozen"].append({"curve": curve(f, seconds), "responses": f["responses"]})
    if verbose:
        for cond, runs in results.items():
            first = np.nanmean([r["curve"][0]["share"] for r in runs])
            last = np.nanmean([r["curve"][-1]["share"] for r in runs])
            print(f"{cond:10s}: '{target}' share {first:.3f} -> {last:.3f}  "
                  f"(smiles {[r['responses'] for r in runs]})")
    if plot:
        _plot(results, target, plot)
    return results


def _plot(results: dict[str, Any], target: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for cond, color in (("contingent", "#2a7"), ("yoked", "#c84"), ("frozen", "#999")):
        runs = results[cond]
        x = [(c["t0"] + c["t1"]) / 120.0 for c in runs[0]["curve"]]
        ys = np.array([[c["share"] for c in r["curve"]] for r in runs], dtype=float)
        ax.plot(x, np.nanmean(ys, axis=0), "-o", color=color, label=cond)
        for y in ys:
            ax.plot(x, y, color=color, alpha=0.25, lw=1)
    ax.axvline(2.0, color="k", lw=0.6, ls=":")
    ax.set(title=f"share of syllables starting with '{target}'", xlabel="minutes (smiles start at 2)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
