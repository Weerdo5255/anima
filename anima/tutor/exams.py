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


def summarize(name: str, res: dict[str, Any]) -> str:
    if name == "forage":
        ch = res.get("choices", {})
        parts = [f"{lab} {ch.get(k, {}).get(act, 0)}/{sum(ch.get(k, {}).values()) or 0}"
                 for k, lab, act in (("90", "left", "left"), ("0", "ahead", "forward"), ("-90", "right", "right"))]
        return f"score {res['score']:.2f} (chance 0.33; correct moves {', '.join(parts)})"
    if name == "ask":
        return (f"says the target sound in {res['hungry']:.0%} of syllables when hungry vs "
                f"{res['sated']:.0%} when sated (n={res['n_hungry']}/{res['n_sated']})")
    return str(res)


EXAMS: dict[str, Callable[..., dict[str, Any]]] = {"forage": forage_exam, "ask": ask_exam}

__all__ = ["EXAMS", "forage_exam", "ask_exam", "hold", "share", "summarize"]
