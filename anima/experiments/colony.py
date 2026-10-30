"""Experiment: does selecting the day's best creature make a better colony?

Three colonies per seed, identical until the first evening: four young
creatures with varied genes share an arena (``anima.evolve``) for several
days of 10 minutes.

* ``evolving`` — at the end of every day in which one creature ate clearly
  more than the others (25% and 0.5 energy more than their median), it and
  three variations of it (copies of its brain with mutated genes, born next
  to it) make up the next day's colony.
* ``random`` — the control for selection: the same turnover on the same
  rule, but the parent of the next day is a creature picked at random, not
  the best.  Whatever copying brains, mutating genes and being born next to
  a parent do by themselves shows up here too.
* ``static`` — the same creatures live on day after day; nobody is ever
  replaced (they still learn).

Measured each day: the food every creature ate (its fitness), how much of it
came from feasts shared with another creature, how many meals were
contested, how much of its waking time it spent near another creature — and,
read from its synapses at the end of the day, how strongly it steers toward
the smell of food and toward the others (``steering``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..evolve import Colony, ColonyConfig, GENE_NAMES
from ..runtime.agent import Agent

NEVER = 1e9  # a margin nobody can beat: no selection
POLICIES = ("evolving", "random", "static")


def steering(agent: Agent, source: str = "sense.bearing") -> float:
    """How strongly a creature's synapses turn it toward what ``source``
    reports: the mean weight from the bearing channels on its left to "left"
    minus to "right", plus the same for its right, relative to the largest
    weight allowed.  0 = no preference; ``sense.bearing`` is the smell of
    food, ``sense.kin`` the others."""
    net = agent.brain.net
    name = f"{source}->motor"
    if name not in net.projections or "motor" not in agent.brain.outputs:
        return float("nan")
    proj = net.projections[name]
    pre, post = net.populations[proj.pre], net.populations[proj.post]
    W = np.where(net.M[proj.pre_slice, proj.post_slice], net.W[proj.pre_slice, proj.post_slice], np.nan)
    m = np.nanmean(W.reshape(pre.channels, pre.size // pre.channels, post.channels, post.size // post.channels),
                   axis=(1, 3)) / proj.w_max
    labels = agent.brain.outputs["motor"].labels
    left, right = labels.index("left"), labels.index("right")
    prefs = np.linspace(-180.0, 180.0, pre.channels, endpoint=False)  # bearing channels, + = to the left
    on_left, on_right = prefs > 10.0, prefs < -10.0
    return float((m[on_left, left] - m[on_left, right]).mean() + (m[on_right, right] - m[on_right, left]).mean())


class _Measured(Colony):
    """A colony that reads its creatures' steering at the end of every day
    (before anyone is replaced).  Its creatures must live in this process."""

    def end_of_day(self) -> dict[str, Any]:
        steer = {cid: {"food": round(steering(self.agent(i), "sense.bearing"), 4),
                       "kin": round(steering(self.agent(i), "sense.kin"), 4)}
                 for i, cid in enumerate(self.members)}
        report = super().end_of_day()
        report["steering"] = steer
        self.save()  # (so colony.json has it too)
        return report


def run_colony(seed: int = 1, days: int = 4, policy: str = "evolving", day_length_s: float = 600.0,
               population: int = 4, cortex: int = 600, keep: str | Path | None = None,
               verbose: bool = False) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    tmp = Path(tempfile.mkdtemp(prefix="anima-colony-"))
    cfg = ColonyConfig(population=population, seed=seed, cortex=cortex, day_length_s=day_length_s,
                       margin=NEVER if policy == "static" else 0.25,
                       selection="random" if policy == "random" else "best", thoughts=False)
    col = _Measured.create(tmp / "colony", cfg, printer=(print if verbose else None))
    try:
        col.run(days=days, report_every_h=0)
    finally:
        col.close()
    out = {"seed": seed, "policy": policy, "days": days, "day_length_s": day_length_s,
           "history": col.history, "records": col.records}
    if keep is not None:
        dest = Path(keep)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "colony" / "colony.json", dest / f"colony_{policy}_s{seed}.json")
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def daily(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-day summary of one colony."""
    rows = []
    for rep in run["history"]:
        cs = rep["creatures"]
        fit = [c["fitness"] for c in cs]
        food = sum(fit)
        feast_energy = sum(c["feasts"] for c in cs) * 0.4
        meals = sum(c["meals"] for c in cs)
        steer = rep.get("steering", {})
        energies = [c["energy_mean"] for c in cs if c["energy_mean"] is not None]
        rows.append({
            "day": rep["day"], "mean_fitness": float(np.mean(fit)), "best": float(np.max(fit)),
            "worst": float(np.min(fit)), "feasts": rep["arena"]["feasts"],
            "feast_share": feast_energy / food if food else 0.0,
            "contested_share": rep["arena"]["contested"] / meals if meals else 0.0,
            "withered": rep["arena"].get("withered", 0),
            "near_others": float(np.mean([c["near_others"] for c in cs])),
            "energy_mean": float(np.mean(energies)) if energies else None,
            "steer_food": float(np.mean([s["food"] for s in steer.values()])) if steer else None,
            "steer_kin": float(np.mean([s["kin"] for s in steer.values()])) if steer else None,
            "generations": sorted({c["generation"] for c in cs}),
            "turnover": bool(rep["decision"]["stands_out"]),
            "parent": rep.get("next", {}).get("champion"),
            "food": {c["id"]: c["fitness"] for c in cs},
            "decision": rep["decision"]})
    return rows


def repeatability(res: dict[str, Any], policy: str = "static") -> dict[str, Any]:
    """Can selection work?  Only if a creature's food on one day says
    something about the next.  Pairs (food on day d, food on day d+1) of the
    same creature in the ``policy`` colonies (static: nobody is replaced, so
    every creature has every day), their correlation, and how often the
    day's best ate more than the others' mean the next day."""
    pairs, best_next = [], []
    for r in res["runs"]:
        if r["policy"] != policy:
            continue
        for a, b in zip(r["daily"], r["daily"][1:]):
            fa, fb = a["food"], b["food"]
            pairs += [(fa[c], fb[c]) for c in fa if c in fb]
            best = max(fa, key=fa.get)
            others = [fb[c] for c in fb if c != best]
            if best in fb and others:
                best_next.append(bool(fb[best] > np.mean(others)))
    x = np.array(pairs, dtype=float)
    r = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if len(x) > 2 and x.std(axis=0).min() > 0 else None
    return {"policy": policy, "n_pairs": len(pairs), "pearson_r": None if r is None else round(r, 3),
            "pairs": [[float(u), float(v)] for u, v in pairs],
            "best_ate_more_than_the_others_next_day": int(sum(best_next)), "of_days": len(best_next)}


def genes_by_day(run: dict[str, Any]) -> list[dict[str, float]]:
    """Mean genes of the creatures alive on each day."""
    out = []
    recs = run["records"]
    for rep in run["history"]:
        ids = [c["id"] for c in rep["creatures"]]
        out.append({g: float(np.mean([recs[i]["genes"][g] for i in ids])) for g in GENE_NAMES})
    return out


def _job(args: tuple[int, int, str, float, int, int, str | None]) -> dict[str, Any]:
    seed, days, policy, day_s, pop, cortex, keep = args
    return run_colony(seed, days, policy, day_s, pop, cortex, keep)


def main(days: int = 4, seeds: tuple[int, ...] = (1, 2), day_length_s: float = 600.0, population: int = 4,
         cortex: int = 600, workers: int = 1, keep: str | None = None, plot: str | None = None,
         policies: tuple[str, ...] = POLICIES, verbose: bool = True) -> dict[str, Any]:
    jobs = [(s, days, p, day_length_s, population, cortex, keep) for s in seeds for p in policies]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(_job, jobs))
    else:
        runs = [_job(j) for j in jobs]
    settings = {"days": days, "day_length_s": day_length_s, "population": population, "cortex": cortex,
                "seeds": list(seeds), "policies": list(policies)}
    return _result(runs, settings, plot, verbose)


def collect(keep: str | Path, plot: str | None = None, verbose: bool = True) -> dict[str, Any]:
    """The same report for colonies run earlier with ``keep`` (their
    ``colony_<policy>_s<seed>.json`` files), without running them again — so
    a long experiment can be run in parts."""
    runs, cfgs = [], []
    for f in sorted(Path(keep).glob("colony_*_s*.json")):
        policy, seed = f.stem[len("colony_"):].rsplit("_s", 1)
        data = json.loads(f.read_text(encoding="utf-8"))
        runs.append({"seed": int(seed), "policy": policy, "history": data["history"], "records": data["records"]})
        cfgs.append(data["config"])
    if not runs:
        raise FileNotFoundError(f"no colony_<policy>_s<seed>.json files in {keep}")
    order = {p: k for k, p in enumerate(POLICIES)}
    runs.sort(key=lambda r: (r["seed"], order.get(r["policy"], 99)))
    settings = {"days": max(len(r["history"]) for r in runs), "day_length_s": cfgs[0]["day_length_s"],
                "population": cfgs[0]["population"], "cortex": cfgs[0]["cortex"],
                "seeds": sorted({r["seed"] for r in runs}),
                "policies": sorted({r["policy"] for r in runs}, key=lambda p: order.get(p, 99))}
    return _result(runs, settings, plot, verbose)


def _result(runs: list[dict[str, Any]], settings: dict[str, Any], plot: str | None, verbose: bool) -> dict[str, Any]:
    res: dict[str, Any] = {"runs": [{"seed": r["seed"], "policy": r["policy"], "daily": daily(r),
                                     "genes": genes_by_day(r)} for r in runs], "settings": settings}
    res["summary"] = summarize(res)
    if "static" in settings["policies"]:
        res["repeatability"] = repeatability(res)
    if verbose:
        print(report(res))
    if plot:
        _plot(res, plot)
    return res


def summarize(res: dict[str, Any]) -> dict[str, Any]:
    """Per policy: mean over seeds, by day, and over the days after the first
    (day 1 is the same for all three)."""
    out = {}
    for policy in res["settings"]["policies"]:
        runs = [r for r in res["runs"] if r["policy"] == policy]
        if not runs:
            continue
        by_day = np.mean([[x["mean_fitness"] for x in r["daily"]] for r in runs], axis=0)

        def mean_of(key: str) -> list[float | None]:
            vals = [[x[key] for x in r["daily"]] for r in runs]
            if any(v is None for row in vals for v in row):
                return [None] * len(vals[0])
            return [round(float(v), 4) for v in np.mean(vals, axis=0)]
        out[policy] = {"mean_fitness_by_day": [round(float(v), 3) for v in by_day],
                       "mean_fitness_after_day1": round(float(np.mean(by_day[1:])), 3) if len(by_day) > 1 else None,
                       "steer_food_by_day": mean_of("steer_food"), "steer_kin_by_day": mean_of("steer_kin"),
                       "near_others_by_day": mean_of("near_others"), "energy_by_day": mean_of("energy_mean"),
                       "turnovers": int(sum(x["turnover"] for r in runs for x in r["daily"][:-1]))}
    return out


def report(res: dict[str, Any]) -> str:
    lines = []
    for r in res["runs"]:
        d = r["daily"]
        lines.append(f"seed {r['seed']} {r['policy']:<8} mean food by day: "
                     + " ".join(f"{x['mean_fitness']:.2f}" for x in d)
                     + " | best: " + " ".join(f"{x['best']:.2f}" for x in d)
                     + " | turnovers: " + "".join("x" if x["turnover"] and r["policy"] != "static" else "."
                                                  for x in d)
                     + " | feasts: " + " ".join(str(x["feasts"]) for x in d)
                     + " | near others: " + " ".join(f"{x['near_others']:.0%}" for x in d)
                     + " | steer food/kin: " + " ".join(
                         "-" if x["steer_food"] is None else f"{x['steer_food']:.2f}/{x['steer_kin']:.2f}"
                         for x in d))
    for policy, s in res.get("summary", summarize(res)).items():
        lines.append(f"{policy:<8} mean over seeds, by day: " + " ".join(f"{v:.2f}" for v in s["mean_fitness_by_day"])
                     + (f"   (days 2+: {s['mean_fitness_after_day1']:.2f})"
                        if s["mean_fitness_after_day1"] is not None else ""))
    rep = res.get("repeatability")
    if rep and rep["pearson_r"] is not None:
        lines.append(f"one creature's food, one day vs the next ({rep['policy']} colonies, {rep['n_pairs']} pairs): "
                     f"r = {rep['pearson_r']:+.2f}; the day's best ate more than the others the next day on "
                     f"{rep['best_ate_more_than_the_others_next_day']} of {rep['of_days']} days")
    return "\n".join(lines)


#: figure colours: categorical slots 1-3 of the validated default palette (anima's figures), text and chrome inks
COLORS = {"evolving": "#2a78d6", "random": "#eb6834", "static": "#1baf7a"}
LABELS = {"evolving": "evolving (the best is the parent)", "random": "random parent", "static": "static"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _plot(res: dict[str, Any], path: str) -> None:
    """Food per creature by day, one panel per seed (the colonies are one line
    until they first differ), and whether a creature's food one day predicts
    the next."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = res["settings"]["seeds"]
    rep = res.get("repeatability")
    n = len(seeds) + (1 if rep and rep["n_pairs"] else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    lo = min(x["mean_fitness"] for r in res["runs"] for x in r["daily"])
    hi = max(x["mean_fitness"] for r in res["runs"] for x in r["daily"])
    for i, (ax, seed) in enumerate(zip(axes, seeds)):
        runs = {r["policy"]: r for r in res["runs"] if r["seed"] == seed}
        series = {p: [x["mean_fitness"] for x in r["daily"]] for p, r in runs.items()}
        days = np.arange(1, len(next(iter(series.values()))) + 1)
        same = next((k for k in range(len(days)) if len({round(v[k], 6) for v in series.values()}) > 1), len(days))
        if same > 0:  # one colony until the first new generation
            ax.plot(days[:same], next(iter(series.values()))[:same], color=MUTED, lw=2, marker="o", ms=7,
                    mec=SURFACE, mew=1.5, solid_capstyle="round", zorder=3,
                    label="one colony until they first differ" if i == 0 else None)
        if same < len(days):
            ax.axvline(same + 0.5, color=GRID, lw=1, zorder=1)
            ax.text(same + 0.55, hi + 0.12, "new generation", color=INK2, fontsize=7.5, va="bottom")
            ends = []
            for p in POLICIES:
                if p not in series:
                    continue
                k0 = max(same - 1, 0)
                ax.plot(days[k0:], series[p][k0:], color=COLORS[p], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                        solid_capstyle="round", zorder=4, label=LABELS[p])
                ends.append((series[p][-1], p))
            ends.sort()
            placed: list[float] = []
            for y, p in ends:  # direct labels, kept apart without stacking far from their lines
                y_text = max(y, placed[-1] + 0.11) if placed else y
                placed.append(y_text)
                ax.text(days[-1] + 0.12, y_text, p if p != "random" else "random", color=INK2, fontsize=8,
                        va="center")
        ax.set_title(f"seed {seed}", color=INK, fontsize=10, loc="left")
        ax.set_xticks(days)
        ax.set_xlim(days[0] - 0.25, days[-1] + 0.85)
        ax.set_ylim(lo - 0.25, hi + 0.35)
        ax.set_xlabel("day", color=INK2, fontsize=9)
    axes[0].set_ylabel("food eaten per creature (energy)", color=INK2, fontsize=9)
    handles, labels = [], []
    for ax in axes[:len(seeds)]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    order = [LABELS[p] for p in POLICIES] + ["one colony until they first differ"]
    pairs = sorted(zip(handles, labels), key=lambda hl: order.index(hl[1]) if hl[1] in order else 99)
    handles, labels = [h for h, _ in pairs], [lab for _, lab in pairs]
    if rep and rep["n_pairs"]:
        ax = axes[-1]
        xy = np.array(rep["pairs"])
        top = float(xy.max()) + 0.3
        ax.plot([0, top], [0, top], color=AXIS, lw=1, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=46, color=COLORS[rep["policy"]], edgecolors=SURFACE, linewidths=1.5,
                   zorder=3)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal", adjustable="box")
        r_txt = "" if rep["pearson_r"] is None else " (r = " + f"{rep['pearson_r']:+.2f})".replace("-", "\u2212")
        ax.set_title(f"one creature, one day vs the next{r_txt}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("food on a day", color=INK2, fontsize=9)
        ax.set_ylabel("food the next day", color=INK2, fontsize=9)
        ax.text(0.04, 0.96, f"{rep['policy']} colonies, {rep['n_pairs']} pairs", transform=ax.transAxes,
                color=INK2, fontsize=8, va="top")
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, labelcolor=INK2)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
                   labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(main(days=1, seeds=(1,), day_length_s=120.0, population=3, cortex=200), default=float)[:500])
