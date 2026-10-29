"""Experiment: does a teacher make a difference?

Two creatures with the same brain seed live in the same world for the same
length of time.  One is raised by a tutor through the four-lesson
curriculum (the scripted teacher by default — its decisions go through the
same tools and validation as an LLM teacher's; pass ``teacher=`` an
``OllamaTeacher`` to use a local model).
The other lives alone with the default world: natural food, nobody to talk
to.  Both keep their plasticity — the untutored creature learns whatever its
life teaches it.

Then both live on alone until 9 am the next morning and sit the same exam
(plasticity paused, nothing is learned during it):

forage   food held 90 deg left / ahead / 90 deg right; fraction of correct moves
babble   two minutes of free babbling: share of syllables starting with "m"
ask      90 s held hungry and 90 s held sated: share of "m" in each
echo     a partner says 30 random syllable pairs (never the target sound); does
         its reply start with the same consonant?  (chance = how often it
         uses that consonant anyway)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..brain.presets import CONSONANTS, VOWELS
from ..runtime.agent import Agent
from .foraging import PROBE_BEARINGS, correct_rate, probe_policy


# ---------------------------------------------------------------------- the exam
def _hold(agent: Agent, seconds: float, energy: float | None = None, say: str | None = None) -> list[str]:
    """Live ``seconds`` with plasticity paused; return the syllables produced."""
    got: list[str] = []
    fn = lambda e: got.append(e["text"])  # noqa: E731
    agent.on("syllable", fn)
    if say:
        agent.hear(say)
    for _ in range(int(seconds * 1000 / agent.tick_ms)):
        if energy is not None:
            agent.body.energy = energy
        agent.tick()
    agent.off("syllable", fn)
    return got


def _wake(agent: Agent, limit_s: float = 900.0) -> None:
    end = agent.t_ms + limit_s * 1000
    while not agent.brain.awake and agent.t_ms < end:
        agent.tick()


def until_morning(agent: Agent, hour: float = 9.0) -> None:
    """Live on (alone) until its clock next shows ``hour`` — a rested, comparable time of day for an exam."""
    d = agent.brain.drives
    wait = ((hour / 24.0 - d.phase) % 1.0) * d.cfg.day_length_s
    agent.run(wait)
    _wake(agent)


def exam(agent: Agent, target: str = "m", seed: int = 0, echo_trials: int = 30) -> dict[str, Any]:
    """Measure the five skills with plasticity paused (state is not restored)."""
    rng = np.random.default_rng(seed)
    _wake(agent)
    net = agent.brain.net
    was = net.plasticity_enabled
    net.plasticity_enabled = False
    w = agent.world
    if hasattr(w, "set_wild"):
        w.set_wild(True)
    try:
        probes = {b: probe_policy(agent, b, n_actions=30) for b in PROBE_BEARINGS}
        free = _hold(agent, 120.0, energy=0.6)
        hungry = _hold(agent, 90.0, energy=0.3)
        sated = _hold(agent, 90.0, energy=0.9)
        share = lambda xs: float(np.mean([s.startswith(target) for s in xs])) if xs else float("nan")  # noqa: E731
        cons = [c for c in CONSONANTS if c not in ("h", target)]  # as in the echo lesson
        hits, chance, answered = 0, [], 0
        usage = {c: np.mean([s.startswith(c) for s in free]) if free else 0.0 for c in cons}
        for _ in range(echo_trials):
            c, v = str(rng.choice(cons)), str(rng.choice(VOWELS))
            reply: list[str] = []
            fn = lambda e, out=reply: out.append(e["text"])  # noqa: E731
            agent.on("syllable", fn)
            agent.hear(f"{c}{v} {c}{v}")
            # wait for the words to be heard, then up to 3 s for the reply
            heard_end = None
            for _ in range(int(8000 / agent.tick_ms)):
                agent.body.energy = 0.6
                agent.tick()
                pending = agent.body.ear.external_pending if agent.body.ear else False
                if heard_end is None and not pending:
                    heard_end = agent.t_ms
                    reply.clear()  # syllables before it finished listening do not count
                if heard_end is not None and (reply or agent.t_ms - heard_end > 3000):
                    break
            agent.off("syllable", fn)
            if reply:
                answered += 1
                hits += reply[0].startswith(c)
            chance.append(usage[c])
            _hold(agent, float(rng.uniform(1.0, 3.0)), energy=0.6)
        return {
            "forage": correct_rate(probes), "probes": probes,
            "babble_share": share(free), "n_free": len(free),
            "ask_hungry": share(hungry), "ask_sated": share(sated),
            "n_hungry": len(hungry), "n_sated": len(sated),
            "echo_rate": hits / max(1, answered), "echo_chance": float(np.mean(chance)),
            "echo_answered": answered, "echo_trials": echo_trials,
        }
    finally:
        net.plasticity_enabled = was


# ---------------------------------------------------------------------- lives
def _energy_log(agent: Agent) -> tuple[list[float], Any]:
    """Record energy once a second while awake (to judge how well it keeps itself fed)."""
    energy: list[float] = []

    def hook(a: Agent) -> None:
        if a.brain.awake and a._ticks % 100 == 0:
            energy.append(a.body.energy)

    agent.add_tick_hook(hook)
    return energy, hook


def run_tutored(seed: int = 1, max_minutes: float = 120.0, teacher: Any = None,
                verbose: bool = False, checkpoint_dir: str | None = None) -> dict[str, Any]:
    from ..tutor import ScriptedTeacher, Tutor
    agent = Agent.create(seed=seed)
    energy, hook = _energy_log(agent)
    tutor = Tutor(agent, teacher or ScriptedTeacher(seed=seed), verbose=False,
                  printer=print if verbose else None)
    t0 = time.time()
    trace = []
    k = saved = 0
    while agent.t_ms < max_minutes * 60_000 and not tutor.curriculum.finished:
        agent.run(tutor.think_every_s)
        tutor.think_once()
        k += 1
        if k % 4 == 0:  # about every 30 s
            r = tutor.meter.report().compact()
            trace.append({"t": agent.t_ms / 1000.0, "lesson": tutor.curriculum.current.name,
                          "awake": agent.brain.awake, "energy": round(agent.body.energy, 3), **r})
        if verbose and k % 15 == 0:
            print(f"{tutor.progress_line()}  (seed {seed}, {agent.t_ms / 60000:.1f} min, "
                  f"wall {time.time() - t0:.0f}s)", flush=True)
        if checkpoint_dir and len(tutor.milestones) > saved:
            saved = len(tutor.milestones)
            stem = f"{checkpoint_dir}/seed{seed}_after_{tutor.milestones[-1]['graduated']}"
            agent.save(stem + ".npz")
            tutor.save(stem + ".tutor.json")
    lived_s = agent.t_ms / 1000.0
    if verbose:
        print(tutor.report())
    tutor.detach()  # from now on it is alone, in the normal world
    until_morning(agent)  # the exam is taken at 9 am the next morning
    alone = float(np.mean(energy[-600:])) if energy else float("nan")  # its last ten awake minutes
    agent.remove_tick_hook(hook)
    result = exam(agent, tutor.target, seed=seed)
    result.update({"seed": seed, "condition": "tutored", "lived_s": lived_s, "exam_at_s": agent.t_ms / 1000.0,
                   "wall_s": time.time() - t0,
                   "milestones": tutor.milestones, "graduated": tutor.curriculum.graduated,
                   "finished": tutor.curriculum.finished, "trace": trace, "alone_energy": alone,
                   "food_given": tutor.toolkit.food_given, "exams_during": tutor.exams})
    return result


def run_untutored(seed: int = 1, seconds: float = 3600.0) -> dict[str, Any]:
    agent = Agent.create(seed=seed)
    energy, hook = _energy_log(agent)
    t0 = time.time()
    agent.run(seconds)
    until_morning(agent)
    alone = float(np.mean(energy[-600:])) if energy else float("nan")
    agent.remove_tick_hook(hook)
    result = exam(agent, "m", seed=seed)
    result.update({"seed": seed, "condition": "untutored", "lived_s": seconds, "exam_at_s": agent.t_ms / 1000.0,
                   "wall_s": time.time() - t0, "alone_energy": alone})
    return result


def summarize(results: list[dict[str, Any]]) -> str:
    lines = ["condition   seed  lived  forage  'm' share  'm' hungry/sated  echo (chance)"]
    for r in results:
        lines.append(f"{r['condition']:<11} {r['seed']:>4}  {r['lived_s'] / 60:4.0f}m   {r['forage']:.2f}     "
                     f"{r['babble_share']:.2f}       {r['ask_hungry']:.2f} / {r['ask_sated']:.2f}       "
                     f"{r['echo_rate']:.2f} ({r['echo_chance']:.2f})")
    return "\n".join(lines)


def main(seeds: tuple[int, ...] = (1, 2), max_minutes: float = 120.0, plot: str | None = None,
         verbose: bool = True, teacher: Any = None) -> dict[str, Any]:
    tutored = [run_tutored(s, max_minutes, teacher=teacher, verbose=verbose) for s in seeds]
    untutored = [run_untutored(s, r["lived_s"]) for s, r in zip(seeds, tutored)]
    res = {"tutored": tutored, "untutored": untutored}
    if verbose:
        print(summarize(tutored + untutored))
    if plot:
        _plot(res, plot)
    return res


def _plot(res: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tut, unt = res["tutored"], res["untutored"]
    metrics = [("forage", "finds food\n(correct moves)"), ("babble_share", "says 'm'\n(share of syllables)"),
               ("echo_rate", "copies your\nconsonant")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.36
    for k, (rs, color, label) in enumerate(((unt, "#9a9a9a", "lived alone"), (tut, "#1f9d6b", "tutored"))):
        xs = x + (k - 0.5) * w
        vals = np.array([[r[m] for m, _ in metrics] for r in rs])
        ax.bar(xs, vals.mean(0), w * 0.94, color=color, label=label)
        for row in vals:
            ax.plot(xs, row, "o", color="k", ms=3, alpha=0.6)
        ch = np.mean([r["echo_chance"] for r in rs])  # chance for each group's own babbling
        ax.plot([xs[2] - w / 2, xs[2] + w / 2], [ch, ch], color="k", lw=1.0, ls=":")
    ax.plot([x[0] - w, x[0] + w], [1 / 3, 1 / 3], color="k", lw=1.0, ls=":")
    ax.text(x[0] + w, 1 / 3, " chance", fontsize=7, va="center")
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set(ylim=(0, 1), ylabel="exam score (plasticity off)", title="the same exam, the morning after")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    from ..tutor.curriculum import LESSON_NAMES
    names = list(LESSON_NAMES)
    for r, color in zip(tut, ("#1f9d6b", "#0b5d3e", "#57c49a")):
        ts = [0.0] + [g["t"] / 60.0 for g in r["graduated"]]
        ax.step(ts, range(len(ts)), where="post", color=color, lw=1.8, label=f"seed {r['seed']}")
        ax.plot(ts[1:], range(1, len(ts)), "o", color=color, ms=4)
    ax.set_yticks(range(len(names) + 1), ["start"] + [f"passed {n}" for n in names])
    ax.set(xlabel="minutes of its life (nights included)", title="graduation timeline")
    ax.legend(frameon=False, loc="lower right")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
