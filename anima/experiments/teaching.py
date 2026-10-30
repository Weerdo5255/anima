"""Experiment: what makes it learn to repeat a teacher's words?

A teacher says a short word every few seconds ("ba ba", "ki ki", ... a
different consonant each time, never "h") while the creature lives in its
arena, kept a little hungry as in the tutor's "Copy me" lesson.  What
happens after its reply differs:

``none``       nothing: the teacher talks but never rewards (being talked to
               still relieves loneliness, which is rewarding in itself)
``smile``      a smile when its first syllable starts with the consonant it
               heard — anima 0.5's ``reward_echo`` rule
``pleasure``   graded pleasure — anima 0.6's ``reward_imitation`` rule: 0.3
               for the right first sound, 0.5 for the right syllable, 1.0 for
               the whole word, while it recalls the reply it is rewarded for
``no-recall``  the same pleasure, but it does not recall its reply

and two runs of a simulated *person* teaching it in the chat, whose verdict
comes late, as a person's does: its reply shows up in the chat once it has
fallen silent, and the person needs a moment to read it and type ``/pleasure``
(it repeated something) or ``/pain`` (it did not) — 2.5-4.5 s after its
reply ended:

``person``            delayed pleasure and pain, for its last words at that
                      moment (``/pleasure speech``); it recalls them
``person-reply``      the same, for its answer to the word, as the chat's ``+``
                      and ``-`` give (they follow the answer to what you last
                      said, even if it has babbled on since)
``person-no-recall``  as ``person``, with no recall

and, for comparison, ``old-mirror`` (``pleasure`` with the hearing->speaking
mirror learning as in anima 0.5: all the time, at a fixed rate, with no echo
masking) and ``old-mirror-none``.

The teacher's choice of words is the ``strategy``: ``focus`` (default) keeps
saying the same word until the creature has repeated at least its first
sound twice (or eight times in vain), then moves on to the next consonant —
as parents do ("mama ... mama ... mama!"); ``rotate`` says a different
consonant every time.

Every five minutes a clone sits the imitation exam (plasticity off: the
examiner says 44 words, every consonant but "h" four times; how often does
its reply repeat the first sound, the syllable, the whole word, against
chance).  Teaching lasts 10 minutes after a first minute alone, so that it
all happens in the creature's morning: with the default 20-minute day it
falls asleep about 13 minutes after it is born, and a clone woken for an
exam in the middle of its night answers differently (a pilot with 15
minutes saw every variant drop at the last exam).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..brain.presets import CONSONANTS, VOWELS
from ..runtime.agent import Agent
from ..tutor.exams import imitate_exam
from ..tutor.skills import LEVELS, SkillMeter
from ..tutor.toolkit import Rule, TeachingToolkit

VARIANTS: dict[str, dict[str, Any]] = {
    "none": {"rule": None, "reinforcer": "pleasure", "recall": True},
    "smile": {"rule": "reward_echo", "reinforcer": "smile", "recall": True},
    "pleasure": {"rule": "reward_imitation", "reinforcer": "pleasure", "recall": True},
    "no-recall": {"rule": "reward_imitation", "reinforcer": "pleasure", "recall": False},
    "person": {"rule": "person", "reinforcer": "pleasure", "recall": True},
    "person-reply": {"rule": "person", "reinforcer": "pleasure", "recall": True, "credit": "reply"},
    "person-no-recall": {"rule": "person", "reinforcer": "pleasure", "recall": False},
    # the hearing->speaking mirror as in anima 0.5: learning all the time, at a fixed rate, no echo masking
    "old-mirror": {"rule": "reward_imitation", "reinforcer": "pleasure", "recall": True, "old_mirror": True},
    "old-mirror-none": {"rule": None, "reinforcer": "pleasure", "recall": True, "old_mirror": True},
}


class SimulatedPerson(Rule):
    """A person teaching in the chat: pleasure if its reply repeated something
    of their word, pain if not — typed a few seconds after its reply ended."""

    name = "person"

    def __init__(self, kit: TeachingToolkit, rng: np.random.Generator, read_s: float = 1.0,
                 react_s: tuple[float, float] = (1.5, 3.5), credit: str = "auto") -> None:
        super().__init__(kit)
        self.rng = rng
        self.read_s, self.react_s = read_s, react_s
        self.credit = credit  # "auto": its last words by then; "reply": its answer itself
        self.given = {"pleasure": 0, "pain": 0}

    def on_reply(self, t, trial):
        res = trial.result or {}
        if not res.get("answered"):
            return
        good = {"word": 1.0, "syllable": 0.7, "consonant": 0.5}.get(res["level"], 0.0)
        kind, amount = ("pleasure", good) if good else ("pain", 0.4)
        self.given[kind] += 1
        due = t + self.read_s + float(self.rng.uniform(*self.react_s))
        credit = list(trial.acts) if self.credit == "reply" and trial.acts else "auto"
        self.kit.schedule(due, lambda: self.kit._feel(kind, amount, credit, "person"),
                          f"{kind} {amount:.1f} (late, from the person)")


def run_one(variant: str = "pleasure", seed: int = 1, minutes: float = 10.0, exam_every: float = 5.0,
            words: int = 44, warmup_s: float = 60.0, strategy: str = "focus", verbose: bool = True,
            spec: dict[str, Any] | None = None) -> dict[str, Any]:
    v = VARIANTS[variant]
    t_wall = time.time()
    if strategy not in ("focus", "rotate"):
        raise ValueError("strategy must be 'focus' or 'rotate'")
    agent = Agent.create(seed=seed, spec=spec)
    if v.get("old_mirror"):
        _old_mirror(agent)
    agent.run(seconds=warmup_s)
    agent.brain.hedonic.cfg.recall = bool(v["recall"])
    meter = SkillMeter(agent)
    kit = TeachingToolkit(agent, meter, rng_seed=seed, reinforcer=v["reinforcer"])
    kit.set_conditions(energy_cap=0.6)
    rng = np.random.default_rng(seed + 100)
    person = None
    if v["rule"] == "person":
        person = SimulatedPerson(kit, np.random.default_rng(seed + 200), credit=v.get("credit", "auto"))
        with kit._lock:
            kit._rules[person.name] = person
    elif v["rule"]:
        kit.set_rule(v["rule"])
    levels: list[tuple[float, str]] = []  # (minute, level) of every reply during teaching
    rank = {lev: i for i, lev in enumerate(LEVELS)}
    current = {"wins": 0}  # replies to the current word that repeated at least its first sound

    class Recorder(Rule):
        name = "recorder"

        def on_reply(self, t, trial):
            if trial.result and trial.result["answered"]:
                levels.append(((t - t0) / 60.0, trial.result["level"]))
                current["wins"] += rank[trial.result["level"]] >= rank["consonant"]

    with kit._lock:
        kit._rules["recorder"] = Recorder(kit)
    cons = [c for c in CONSONANTS if c != "h"]
    order: list[str] = []
    exams: list[dict[str, Any]] = []

    def exam(minute: float) -> None:
        r = imitate_exam(agent, words=words, seed=1000 + seed)
        r.pop("replies", None)
        exams.append({"minute": round(minute, 2), **r})
        if verbose:
            print(f"  [{variant}/{strategy} seed {seed}] minute {minute:4.1f}: repeats first sound {r['consonant']:.2f} "
                  f"(chance {r['consonant_chance']:.2f}), syllable {r['syllable']:.2f} "
                  f"(chance {r['syllable_chance']:.2f}), word {r['word']:.2f}  n={r['n']}/{r['trials']}"
                  f"  ({time.time() - t_wall:.0f}s)", flush=True)

    t0 = agent.t_ms / 1000.0
    exam(0.0)
    next_say = t0 + 2.0
    next_exam = exam_every
    ear, voice = agent.body.ear, agent.body.voice
    end = t0 + minutes * 60.0
    word, tries = None, 0
    while agent.t_ms / 1000.0 < end:
        t = agent.t_ms / 1000.0
        if t >= next_say and not ear.external_pending and not voice.speaking:
            if word is None or strategy == "rotate" or current["wins"] >= 2 or tries >= 8:
                if not order:
                    order = list(rng.permutation(cons))
                syl = order.pop() + str(rng.choice(VOWELS))
                word, tries, current["wins"] = f"{syl} {syl}", 0, 0
            kit._say(word)
            tries += 1
            next_say = t + float(rng.uniform(5.0, 7.0))
        agent.tick()
        if (agent.t_ms / 1000.0 - t0) / 60.0 >= next_exam:
            exam(next_exam)
            next_exam += exam_every
    blocks = []
    for b0 in np.arange(0.0, minutes, exam_every):
        lv = [rank[x] for m, x in levels if b0 <= m < b0 + exam_every]
        blocks.append({"from_minute": float(b0), "replies": len(lv),
                       "consonant": float(np.mean([x >= rank["consonant"] for x in lv])) if lv else None,
                       "syllable": float(np.mean([x >= rank["syllable"] for x in lv])) if lv else None,
                       "word": float(np.mean([x >= rank["word"] for x in lv])) if lv else None})
    kit.detach()
    meter.detach()
    return {"variant": variant, "seed": seed, "strategy": strategy, "minutes": minutes, "exams": exams,
            "during": blocks,
            "feelings": dict(agent.brain.hedonic.counts), "recalls": agent.brain.recalls,
            "person": dict(person.given) if person else None, "wall_s": round(time.time() - t_wall, 1)}


# ----------------------------------------------------------------------------- two closer looks
def _trial(agent: Agent, word: str, heard: list[dict[str, Any]], max_s: float = 6.0) -> Any:
    """Say ``word`` (after a quiet moment) and follow its reply."""
    from ..tutor.skills import WordTrial
    voice, ear = agent.body.voice, agent.body.ear
    for _ in range(300):
        if not voice.speaking:
            break
        agent.tick()
    trial = WordTrial(word, agent.t_ms / 1000.0)
    agent.hear(word)
    seen, was = len(heard), True
    for _ in range(int(max_s * 100)):
        agent.tick()
        t = agent.t_ms / 1000.0
        hearing = ear.external_pending
        if was and not hearing:
            trial.heard_end(t)
        was = hearing
        for e in heard[seen:]:
            trial.syllable(e["t"], e["text"], e.get("act"))
        seen = len(heard)
        if trial.closed or trial.tick(t):
            break
    return trial


def _old_mirror(agent: Agent) -> None:
    """Make the hearing->speaking mirror learn as in anima 0.5: all the time
    (also while it only listens), at a fixed rate, with no echo masking."""
    agent.brain.effects.hebbian_dopamine = 0.0
    agent.brain._mirror_gate = lambda: 1.0
    agent.body.ear.self_masking = 1.0


def _row_contrast(agent: Agent, consonant: str, proj: str = "audio->vocal.C") -> float:
    """Diagonal / mean off-diagonal weight of one consonant's row of the hearing->speaking map."""
    from .imitation import mirror_matrix
    m = mirror_matrix(agent, proj)
    k = CONSONANTS.index(consonant)
    return float(m[k, k] / max(np.delete(m[k], k).mean(), 1e-12))


def practice_one_word(seed: int = 1, word: str = "ba ba", tries: int = 40, reward: bool = True,
                      old_mirror: bool = False, warmup_s: float = 60.0) -> dict[str, Any]:
    """Say the same word ``tries`` times; when the reply repeats at least its
    first sound, give pleasure for it (``reward``).  How do the hearing->
    speaking map's row for that consonant, and the copies, change?
    ``old_mirror``: the mirror learns as in anima 0.5 (also while it only
    listens, at a fixed rate; no echo masking)."""
    agent = Agent.create(seed=seed)
    if old_mirror:
        _old_mirror(agent)
    agent.run(seconds=warmup_s)
    c = word[0]
    before = _row_contrast(agent, c)
    heard: list[dict[str, Any]] = []
    agent.on("syllable", heard.append)
    copies = []
    for _ in range(tries):
        tr = _trial(agent, word, heard)
        ok = bool(tr.result and tr.result["answered"] and tr.result["consonant"])
        copies.append(ok)
        if ok and reward:
            agent.brain.feel("pleasure", 0.7 if tr.result["syllable"] else 0.5, "teacher", tr.acts or "now")
        agent.run(seconds=1.5)
    return {"seed": seed, "word": word, "reward": reward, "old_mirror": old_mirror, "row_before": round(before, 2),
            "row_after": round(_row_contrast(agent, c), 2), "copies_by_ten": [int(sum(copies[i:i + 10]))
                                                                            for i in range(0, tries, 10)]}


def mirror_ceiling(seed: int = 1, age_s: float = 120.0, words: int = 44) -> dict[str, Any]:
    """How well could it copy with a perfect hearing->speaking map?  The
    mirror's weights are redistributed onto the matching channels by hand
    (each articulator neuron keeps its total input), half-way and fully."""
    agent = Agent.create(seed=seed)
    agent.run(seconds=age_s)
    net = agent.brain.net
    out = {"seed": seed, "natural": imitate_exam(agent, words=words, seed=5)}
    for frac in (0.5, 1.0):
        for proj, offset in (("audio->vocal.C", 0), ("audio->vocal.V", len(CONSONANTS))):
            p = net.projections[proj]
            W, M = net.weights(proj).copy(), net.mask(proj)
            cs_pre, cs_post = p.pre_pop.channel_size, p.post_pop.channel_size
            for j in range(p.post_pop.channels):
                cols = slice(j * cs_post, (j + 1) * cs_post)
                budget = W[:, cols].sum(axis=0)
                rows = slice((offset + j) * cs_pre, (offset + j + 1) * cs_pre)
                new = W[:, cols] * (1.0 - frac)
                m = M[rows, cols].astype(float)
                new[rows, :] += m * (frac * budget / np.maximum(m.sum(axis=0), 1.0))[None, :]
                W[:, cols] = new
            net.set_weights(proj, W.astype(np.float32))
        out[f"diagonal_{int(frac * 100)}"] = imitate_exam(agent, words=words, seed=5)
    for v in out.values():
        if isinstance(v, dict):
            v.pop("replies", None)
    return out


def late_shaping(seed: int = 1, recall: bool = True, minutes: float = 10.0, target: str = "m",
                 delay_s: tuple[float, float] = (2.5, 4.5), amount: float = 0.6,
                 block_min: float = 2.0) -> dict[str, Any]:
    """Shaping its babbling as a person in the chat would: whenever one of its
    utterances contains a syllable starting with ``target``, ``/pleasure``
    comes 2.5-4.5 s after the utterance has ended (it shows up in the chat
    once it has been silent for a moment, and then the person needs to read
    and type) — credit "auto", as the chat gives.  With and without recall.
    Returns the share of its syllables that start with ``target``, per block."""
    agent = Agent.create(seed=seed)
    agent.brain.hedonic.cfg.recall = bool(recall)
    rng = np.random.default_rng(seed + 300)
    syll: list[tuple[float, str]] = []
    due: list[float] = []
    agent.on("syllable", lambda e: syll.append((e["t"], e["text"])))
    agent.on("utterance", lambda e: due.append(e["t"] + float(rng.uniform(*delay_s)))
             if any(w.startswith(target) for w in _syllables_of(e["text"])) else None)
    given = 0
    end = minutes * 60.0
    while agent.t_ms / 1000.0 < end:
        agent.tick()
        t = agent.t_ms / 1000.0
        for d in [d for d in due if d <= t]:
            due.remove(d)
            agent.brain.feel("pleasure", amount, "person", "auto")
            given += 1
    blocks = []
    for b0 in np.arange(0.0, minutes, block_min):
        xs = [x for t, x in syll if b0 * 60 <= t < (b0 + block_min) * 60]
        blocks.append(round(float(np.mean([x.startswith(target) for x in xs])), 3) if xs else None)
    return {"seed": seed, "recall": recall, "target": target, "share_by_block": blocks, "pleasures": given,
            "block_minutes": block_min, "syllables": len(syll)}


def _syllables_of(text: str) -> list[str]:
    from ..body.voice import syllabify
    return syllabify(text)


def _shaping_job(args: tuple[int, bool]) -> dict[str, Any]:
    return late_shaping(seed=args[0], recall=args[1])


def _job(args: tuple[str, int, float, float, str]) -> dict[str, Any]:
    variant, seed, minutes, exam_every, strategy = args
    return run_one(variant, seed, minutes, exam_every, strategy=strategy)


def _practice_job(args: tuple[int, bool, bool]) -> dict[str, Any]:
    seed, reward, old = args
    return practice_one_word(seed=seed, reward=reward, old_mirror=old)


def studies(seeds: tuple[int, ...] = (1, 2), workers: int = 1, verbose: bool = True) -> dict[str, Any]:
    """Three closer looks: one word practised 40 times (pleasure for copies
    or not; the v0.6 mirror rule or anima 0.5's); the perfect-map ceiling;
    and shaping its babbling with pleasure that comes late, recall on or off."""
    jobs = [(s, r, o) for o in (False, True) for r in (True, False) for s in seeds]
    shaping_jobs = [(s, r) for r in (True, False) for s in seeds]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            practice = list(ex.map(_practice_job, jobs))
            ceiling = list(ex.map(mirror_ceiling, seeds))
            shaping = list(ex.map(_shaping_job, shaping_jobs))
    else:
        practice = [_practice_job(j) for j in jobs]
        ceiling = [mirror_ceiling(s) for s in seeds]
        shaping = [_shaping_job(j) for j in shaping_jobs]
    if verbose:
        for r in practice:
            print(f"  practice 'ba ba' x40, {'old mirror' if r['old_mirror'] else 'v0.6 mirror'}, "
                  f"{'pleasure for copies' if r['reward'] else 'no reward'}, seed {r['seed']}: 'b' row "
                  f"{r['row_before']:.2f} -> {r['row_after']:.2f}; copies per 10 tries {r['copies_by_ten']}")
        for c in ceiling:
            parts = [f"{k}: first sound {v['consonant']:.2f}, syllable {v['syllable']:.2f}"
                     for k, v in c.items() if isinstance(v, dict)]
            print(f"  ceiling seed {c['seed']}: " + "; ".join(parts))
        for r in shaping:
            print(f"  late pleasure for 'm', recall {'on' if r['recall'] else 'off'}, seed {r['seed']}: share of 'm' "
                  f"per {r['block_minutes']:g} min {r['share_by_block']} ({r['pleasures']} pleasures)")
    return {"practice": practice, "ceiling": ceiling, "shaping": shaping}


def main(minutes: float = 10.0, seeds: tuple[int, ...] = (1, 2, 3), variants: tuple[str, ...] = tuple(VARIANTS),
         exam_every: float = 5.0, workers: int = 1, plot: str | None = None, verbose: bool = True,
         strategy: str = "focus", parts: tuple[str, ...] = ("variants",)) -> dict[str, Any]:
    if "variants" not in parts:
        return studies(seeds, workers, verbose) if {"practice", "ceiling"} & set(parts) else {}
    jobs = [(v, s, minutes, exam_every, strategy) for v in variants for s in seeds]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(_job, jobs))
    else:
        runs = [_job(j) for j in jobs]
    results: dict[str, Any] = {"minutes": minutes, "seeds": list(seeds), "strategy": strategy, "runs": runs,
                               "summary": {}}
    for v in variants:
        rs = [r for r in runs if r["variant"] == v]
        last = [r["exams"][-1] for r in rs]
        first = [r["exams"][0] for r in rs]
        results["summary"][v] = {k: {"before": round(float(np.nanmean([e[k] for e in first])), 3),
                                     "after": round(float(np.nanmean([e[k] for e in last])), 3)}
                                 for k in ("consonant", "consonant_chance", "syllable", "syllable_chance", "word")}
        if verbose:
            s = results["summary"][v]
            print(f"{v:>17}: first sound {s['consonant']['before']:.2f} -> {s['consonant']['after']:.2f} "
                  f"(chance {s['consonant_chance']['after']:.2f}); syllable {s['syllable']['before']:.2f} -> "
                  f"{s['syllable']['after']:.2f} (chance {s['syllable_chance']['after']:.2f}); whole word "
                  f"{s['word']['before']:.2f} -> {s['word']['after']:.2f}", flush=True)
    if plot:
        _plot(results, plot)
    if {"practice", "ceiling"} & set(parts):
        results["studies"] = studies(seeds, workers, verbose)
    return results


#: how each variant reads in the figure (top to bottom), and which mirror rule it used
LABELS = {
    "none": "nothing: the teacher only talks",
    "smile": "a smile if the first sound is right (0.5's reflex)",
    "pleasure": "graded pleasure, with recall (0.6's reflex)",
    "no-recall": "graded pleasure, no recall",
    "person": "a person, 2.5-4.5 s late, for its last words",
    "person-reply": "a person, 2.5-4.5 s late, for its answer",
    "person-no-recall": "a person, 2.5-4.5 s late, no recall",
    "old-mirror": "graded pleasure, with recall",
    "old-mirror-none": "nothing: the teacher only talks",
}
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plot(results: dict[str, Any], path: str) -> None:
    """The last exam of each variant: bars are the mean over seeds, dots the seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [v for v in LABELS if v in results["summary"]]
    rows = {v: [r for r in results["runs"] if r["variant"] == v] for v in names}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(names) + 1.6), sharey=True, facecolor=SURFACE)
    y = np.arange(len(names))[::-1]
    for ax, key, title in zip(axes, ("consonant", "syllable"),
                              ("repeats the first sound of the word", "repeats its first syllable")):
        ax.set_facecolor(SURFACE)
        for yy, v in zip(y, names):
            vals = [r["exams"][-1][key] for r in rows[v]]
            color = ORANGE if VARIANTS[v].get("old_mirror") else BLUE
            ax.barh(yy, float(np.mean(vals)), height=0.56, color=color, zorder=2)
            ax.plot(vals, [yy] * len(vals), "o", ms=4.5, color=SURFACE, mec=INK, mew=0.9, zorder=3)
        before = float(np.mean([r["exams"][0][key] for v in names for r in rows[v]]))
        chance = float(np.mean([r["exams"][-1][f"{key}_chance"] for v in names for r in rows[v]]))
        ax.axvline(chance, color=INK2, lw=1.0, ls=":", zorder=1)
        ax.axvline(before, color=INK2, lw=1.0, ls="--", zorder=1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlim(0, max(0.05, ax.get_xlim()[1]))
        ax.set_xlabel(f"fraction of replies after {results['minutes']:g} minutes (exam, plasticity off)",
                      fontsize=8.5, color=INK2)
    axes[0].set_yticks(y, [LABELS[v] for v in names], fontsize=8.5, color=INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=BLUE, label="its hearing→speaking map learns while it speaks (anima 0.6)"),
               Patch(color=ORANGE, label="... learns all the time (anima 0.5)"),
               Line2D([], [], color=INK2, ls="--", lw=1.0, label="before teaching"),
               Line2D([], [], color=INK2, ls=":", lw=1.0, label="chance")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=4, labelcolor=INK)
    fig.suptitle(f"What follows its replies to a teacher's words — {len(rows[names[0]])} seeds (dots), bars: mean",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


__all__ = ["run_one", "main", "VARIANTS", "SimulatedPerson"]
