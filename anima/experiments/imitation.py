"""Experiment: learning to imitate by listening to itself.

Nobody tells the creature which sound each articulation makes.  While it
babbles alone it hears its own voice ~20 ms after producing each syllable,
and Hebbian synapses from the auditory populations to the vocal ones pick
up that consistent co-activation ("audio /b/ goes with vocal b").  This is
the classic account of how mirror-like sensorimotor mappings arise from
self-observation (Heyes 2010; Westermann & Miranda 2004 for babbling).

Phase 1  babble alone; measure how diagonal the auditory->vocal map becomes.
Phase 2  a partner says a syllable ("dada", "momo", ...); does the creature's
         next syllable start with the consonant it just heard more often than
         chance?  Compared against an identical creature whose auditory->vocal
         map was frozen during phase 1 (so it never learned it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..brain.presets import CONSONANTS, VOWELS
from ..runtime.agent import Agent


def mirror_matrix(agent: Agent, proj: str = "audio->vocal.C") -> np.ndarray:
    """Mean weight from each heard phoneme channel to each articulator channel."""
    net = agent.brain.net
    p = net.projections[proj]
    W, M = net.weights(proj), net.mask(proj)
    pre, post = p.pre_pop, p.post_pop
    rows = [k for k in range(pre.channels)
            if M[k * pre.channel_size:(k + 1) * pre.channel_size].any()]
    out = np.zeros((len(rows), post.channels))
    for a, k in enumerate(rows):
        rs = slice(k * pre.channel_size, (k + 1) * pre.channel_size)
        for j in range(post.channels):
            cs = slice(j * post.channel_size, (j + 1) * post.channel_size)
            m = M[rs, cs]
            out[a, j] = W[rs, cs][m].mean() if m.any() else 0.0
    return out


def diagonal_index(m: np.ndarray) -> float:
    """Mean diagonal weight / mean off-diagonal weight (1 = no mapping)."""
    n = min(m.shape)
    d = np.diag(m[:n, :n]).mean()
    off = (m[:n, :n].sum() - np.trace(m[:n, :n])) / (n * n - n)
    return float(d / max(off, 1e-12))


def run_imitation(babble_s: float = 300.0, test_s: float = 300.0, seed: int = 1,
                  learn_mirror: bool = True, verbose: bool = False) -> dict[str, Any]:
    agent = Agent.create(seed=seed, world="void")
    rules = [agent.brain.net.projections[n].rule for n in ("audio->vocal.C", "audio->vocal.V")]
    saved_rates = [r.rate for r in rules]
    if not learn_mirror:
        for r in rules:
            r.rate = 0.0
    syl: list[tuple[float, str]] = []
    agent.on("syllable", lambda e: syl.append((e["t"], e["text"])))
    m0 = mirror_matrix(agent)
    v0 = mirror_matrix(agent, "audio->vocal.V")
    ticks = int(babble_s * 1000 / agent.tick_ms)
    for _ in range(ticks):
        agent.body.energy = 0.8
        agent.tick()
    m1 = mirror_matrix(agent)
    v1 = mirror_matrix(agent, "audio->vocal.V")
    for r, rate in zip(rules, saved_rates):
        r.rate = rate if learn_mirror else 0.0
    if verbose:
        print(f"  [seed {seed} mirror {'on' if learn_mirror else 'off'}] after {babble_s:.0f}s babbling: "
              f"consonant map diagonal index {diagonal_index(m0):.2f} -> {diagonal_index(m1):.2f}, "
              f"vowels {diagonal_index(v1):.2f}; {len(syl)} syllables", flush=True)
    base_counts = {c: 0 for c in CONSONANTS}
    base_vowels = {v: 0 for v in VOWELS}
    for _, s in syl:
        if s[0] in base_counts:
            base_counts[s[0]] += 1
        base_vowels[s[-1]] += 1
    # ---------------- phase 2: partner speaks, does it echo?
    rng = np.random.default_rng(seed + 50)
    probes = []  # (consonant said, end time, response consonant or None)
    next_t = agent.t_ms / 1000.0 + 3.0
    end = agent.t_ms / 1000.0 + test_s
    current = None
    while agent.t_ms / 1000.0 < end:
        t = agent.t_ms / 1000.0
        if current is None and t >= next_t:
            c = str(rng.choice(CONSONANTS))
            v = str(rng.choice(VOWELS))
            agent.hear(f"{c}{v}{c}{v}")
            current = {"c": c, "v": v, "start": t, "end": None, "resp": None, "resp_v": None}
        agent.body.energy = 0.8
        agent.tick()
        if current is not None:
            ear = agent.body.ear
            if current["end"] is None and t > current["start"] + 0.1 and not ear.external_pending:
                current["end"] = t
            if current["end"] is not None:
                newer = [s for ts, s in syl if ts > current["end"]]
                if newer:
                    current["resp"] = newer[0][0] if newer[0][0] in CONSONANTS else "-"
                    current["resp_v"] = newer[0][-1]
                if newer or t > current["end"] + 2.5:
                    probes.append(current)
                    current = None
                    next_t = t + float(rng.uniform(4.0, 8.0))
    answered = [p for p in probes if p["resp"] is not None]
    matches = sum(p["resp"] == p["c"] for p in answered)
    v_matches = sum(p["resp_v"] == p["v"] for p in answered)
    total_base = max(1, sum(base_counts.values()))
    total_v = max(1, sum(base_vowels.values()))
    # chance of matching = mean over probes of the baseline probability of
    # producing that phoneme while babbling alone
    chance = float(np.mean([base_counts[p["c"]] / total_base for p in answered])) if answered else float("nan")
    chance_v = float(np.mean([base_vowels[p["v"]] / total_v for p in answered])) if answered else float("nan")
    n = len(answered)
    return {
        "diag_before": diagonal_index(m0), "diag_after": diagonal_index(m1),
        "vowel_diag_before": diagonal_index(v0), "vowel_diag_after": diagonal_index(v1),
        "probes": len(probes), "answered": n,
        "echo_rate": matches / n if n else float("nan"), "chance": chance,
        "vowel_echo_rate": v_matches / n if n else float("nan"), "vowel_chance": chance_v,
        "mirror_matrix": m1.tolist(),
    }


def main(seconds: float = 300.0, seeds: tuple[int, ...] = (1, 2), plot: str | None = None,
         verbose: bool = True) -> dict[str, Any]:
    results: dict[str, Any] = {"learned": [], "frozen": []}
    for cond in ("learned", "frozen"):
        for s in seeds:
            r = run_imitation(babble_s=seconds, test_s=seconds, seed=s,
                              learn_mirror=(cond == "learned"), verbose=verbose)
            if verbose:
                print(f"  [{cond} seed {s}] consonant echo {r['echo_rate']:.2f} vs chance "
                      f"{r['chance']:.2f}; vowel echo {r['vowel_echo_rate']:.2f} vs chance "
                      f"{r['vowel_chance']:.2f} ({r['answered']}/{r['probes']} answered)", flush=True)
            results[cond].append(r)
    if plot:
        _plot(results, plot)
    return results


def _plot(results: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    m = np.asarray(results["learned"][0]["mirror_matrix"])
    im = axes[0].imshow(m, cmap="viridis")
    axes[0].set_xticks(range(len(CONSONANTS)), CONSONANTS)
    axes[0].set_yticks(range(len(CONSONANTS)), CONSONANTS)
    axes[0].set(xlabel="articulates", ylabel="hears", title="learned hearing -> speaking map")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    groups = []
    for cond in ("learned", "frozen"):
        runs = results[cond]
        groups.append((f"{cond}\nconsonant", np.mean([r["echo_rate"] for r in runs]),
                       np.mean([r["chance"] for r in runs])))
        groups.append((f"{cond}\nvowel", np.mean([r["vowel_echo_rate"] for r in runs]),
                       np.mean([r["vowel_chance"] for r in runs])))
    x = np.arange(len(groups))
    axes[1].bar(x - 0.2, [g[1] for g in groups], 0.4, color="#1f9d6b", label="matches what it heard")
    axes[1].bar(x + 0.2, [g[2] for g in groups], 0.4, color="#bbb", label="chance")
    axes[1].set_xticks(x, [g[0] for g in groups], fontsize=8)
    axes[1].set(title="its first syllable after you speak", ylim=(0, 0.8))
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
