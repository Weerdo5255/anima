"""Exams: test a skill on a snapshot of the creature, with plasticity off.

A teacher needs to know whether a skill is really there.  Watching natural
behaviour is enough for some skills (which sounds it babbles, whether it
copies a sound it just heard, whether it keeps itself fed), but a poor test
of others: in an arena with several pieces of food the smell usually comes
from several directions at once, so most natural moves say little about
whether it can follow its nose; and a creature that is fed whenever it calls
is rarely hungry for long, so "does it call more when hungry?" is slow to
answer by watching.

For those skills the tutor gives an exam on a **clone** of the creature —
same synapses, same state, plasticity switched off.  The real creature is not
interrupted, moved or trained by being tested, and nothing it does in the
exam can teach it anything.

forage  food held 90 deg left, straight ahead and 90 deg right (as in
        ``anima experiment foraging``); score = fraction of correct moves,
        chance 1/3
ask     alternating 15 s blocks held hungry (energy 0.3) and sated (energy
        0.9): share of syllables starting with the target sound in each
imitate the examiner says 33 short words ("ba ba", "ki ki"), every consonant
        but "h" three times; how often does its reply repeat the first consonant, the
        first syllable, the whole word — against chance (how often its replies
        would match by luck, given which sounds it uses in them)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..experiments.foraging import PROBE_BEARINGS, correct_rate, probe_policy


def _subject(agent: Any) -> Any:
    """A clone ready to be examined: plasticity off, awake."""
    c = agent.clone()
    c.brain.net.plasticity_enabled = False
    if not c.brain.awake:
        c.brain.sleep.force("awake", c.brain.time_s)
    return c


def hold(agent: Any, seconds: float, energy: float | None = None) -> list[str]:
    """Live ``seconds`` (energy held fixed if given); return the syllables it said."""
    got: list[str] = []
    fn = lambda e: got.append(e["text"])  # noqa: E731
    agent.on("syllable", fn)
    try:
        for _ in range(int(seconds * 1000 / agent.tick_ms)):
            if energy is not None:
                agent.body.energy = energy
            agent.tick()
    finally:
        agent.off("syllable", fn)
    return got


def share(syllables: list[str], target: str) -> float:
    return float(np.mean([s.startswith(target) for s in syllables])) if syllables else float("nan")


def forage_exam(agent: Any, n_actions: int = 20, **_: Any) -> dict[str, Any]:
    c = _subject(agent)
    choices = {b: probe_policy(c, b, n_actions=n_actions) for b in PROBE_BEARINGS}
    return {"score": round(correct_rate(choices), 3),
            "choices": {str(b): v for b, v in choices.items()}}


def ask_exam(agent: Any, target: str = "m", block_s: float = 15.0, blocks: int = 3, **_: Any) -> dict[str, Any]:
    c = _subject(agent)
    hungry: list[str] = []
    sated: list[str] = []
    for _ in range(blocks):
        hungry += hold(c, block_s, energy=0.3)
        sated += hold(c, block_s, energy=0.9)
    return {"hungry": round(share(hungry, target), 3), "sated": round(share(sated, target), 3),
            "n_hungry": len(hungry), "n_sated": len(sated)}


def imitate_exam(agent: Any, words: int = 33, seed: int = 0, gap_s: float = 1.0, **_: Any) -> dict[str, Any]:
    """Say ``words`` short reduplicated words to a clone and score its replies."""
    from ..brain.presets import CONSONANTS, VOWELS
    from .skills import LEVELS, WordTrial
    c = _subject(agent)
    rng = np.random.default_rng(seed)
    cons = [x for x in CONSONANTS if x != "h"]
    order = [cons[k % len(cons)] for k in range(words)]
    rng.shuffle(order)
    energy = float(c.body.energy)
    voice, ear = c.body.voice, c.body.ear
    trials: list[dict[str, Any]] = []
    got: list[tuple[float, str]] = []
    c.on("syllable", lambda e: got.append((e["t"], e["text"])))

    def tick() -> None:
        c.body.energy = energy
        c.tick()

    for k, con in enumerate(order):
        # wait for a quiet moment (at most 4 s), as a teacher would
        for _ in range(400):
            if not voice.speaking and c.t_ms / 1000.0 - voice.last_syllable_t / 1000.0 >= gap_s:
                break
            tick()
        syl = con + str(rng.choice(VOWELS))
        word = f"{syl} {syl}"
        trial = WordTrial(word, c.t_ms / 1000.0)
        c.hear(word)
        seen = len(got)
        was = True
        for _ in range(1000):
            tick()
            hearing = ear.external_pending
            if was and not hearing:
                trial.heard_end(c.t_ms / 1000.0)
            was = hearing
            for t, x in got[seen:]:
                trial.syllable(t, x)
            seen = len(got)
            if trial.closed or trial.tick(c.t_ms / 1000.0):
                break
        if trial.result is not None:
            trials.append(trial.result)
    answered = [t for t in trials if t["answered"]]
    out: dict[str, Any] = {"trials": len(trials), "n": len(answered),
                           "replies": [[t["said"], " ".join(t["reply"])] for t in trials]}
    if not answered:
        return {**out, **{k: float("nan") for k in ("consonant", "vowel", "syllable", "word", "consonant_chance",
                                                     "syllable_chance", "word_chance")}}
    rank = {lev: i for i, lev in enumerate(LEVELS)}
    for lev in ("consonant", "syllable", "word"):
        out[lev] = round(float(np.mean([rank[t["level"]] >= rank[lev] for t in answered])), 3)
    out["vowel"] = round(float(np.mean([t["vowel"] for t in answered])), 3)
    first = [t["reply"][0] for t in answered]
    every = [x for t in answered for x in t["reply"]]
    p_syl = {x: first.count(x) / len(first) for x in set(first)}
    p_any = {x: every.count(x) / len(every) for x in set(every)}
    p_con = {}
    for x in first:
        p_con[x[:-1]] = p_con.get(x[:-1], 0.0) + 1.0 / len(first)
    out["consonant_chance"] = round(float(np.mean([p_con.get(t["target"][0][:-1], 0.0) for t in answered])), 3)
    out["syllable_chance"] = round(float(np.mean([p_syl.get(t["target"][0], 0.0) for t in answered])), 3)
    out["word_chance"] = round(float(np.mean([p_syl.get(t["target"][0], 0.0) * float(np.prod(
        [p_any.get(x, 0.0) for x in t["target"][1:]])) for t in answered])), 4)
    return out


def summarize(name: str, res: dict[str, Any]) -> str:
    if name == "forage":
        ch = res.get("choices", {})
        parts = [f"{lab} {ch.get(k, {}).get(act, 0)}/{sum(ch.get(k, {}).values()) or 0}"
                 for k, lab, act in (("90", "left", "left"), ("0", "ahead", "forward"), ("-90", "right", "right"))]
        return f"score {res['score']:.2f} (chance 0.33; correct moves {', '.join(parts)})"
    if name == "ask":
        return (f"says the target sound in {res['hungry']:.0%} of syllables when hungry vs "
                f"{res['sated']:.0%} when sated (n={res['n_hungry']}/{res['n_sated']})")
    if name == "imitate":
        if not res.get("n"):
            return f"no replies to {res.get('trials', 0)} words"
        return (f"repeats the first sound {res['consonant']:.0%} (chance {res['consonant_chance']:.0%}), "
                f"the syllable {res['syllable']:.0%} (chance {res['syllable_chance']:.0%}), "
                f"the whole word {res['word']:.0%} (n={res['n']}/{res['trials']})")
    return str(res)


EXAMS: dict[str, Callable[..., dict[str, Any]]] = {"forage": forage_exam, "ask": ask_exam, "imitate": imitate_exam}

__all__ = ["EXAMS", "forage_exam", "ask_exam", "imitate_exam", "hold", "share", "summarize"]
