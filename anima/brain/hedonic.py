"""Pain and pleasure: two innate feelings, and the most direct way to teach.

Every animal is born able to feel pleasure (a treat, being stroked) and pain
(a pinch, a sting).  Nothing has to be learned for them to matter: they are
*primary reinforcers*.  In the brain they reach the dopamine system the way
food does — pleasure as a burst, pain as a dip below baseline (Schultz 1998;
Ungless et al. 2004 for the dip) — and so they teach: whatever the creature
just did becomes more likely (pleasure) or less likely (pain) the next time
it is in the same situation.

:meth:`Brain.feel` gives the creature one of them.  Four things follow:

1. **A teaching signal.**  For a moment the brain receives a reward (+) or a
   punishment (-).  It goes through the critic like any other reward, so
   the dopamine it causes is a *prediction error*: pleasure it has learned
   to expect in a situation is worth less there.  The intensity (0-1) sets
   how strong the signal is and how long it lasts (0.2-0.7 s); the dopamine
   it releases is several times that of a smile.
2. **It recalls what it just did** (awake replay).  Learning needs the
   reward to come within a second or two of the act (eligibility traces),
   but a person typing, or a language model thinking, takes longer.  Animals
   bridge such delays: when a rat reaches a reward its hippocampus replays,
   in reverse, the path that led there (Foster & Wilson 2006), more so for
   bigger rewards (Ambrose et al. 2016).  Here the creature pauses and
   replays its last words (or its last movement) — the syllables it said
   and what its senses reported when it decided to say them, including the
   echo of the sound it had just heard — while the dopamine is there, so the
   right synapses are credited even when the feeling comes seconds late.
   For the voice, the recall also binds the sound it had heard to the sound
   it made, in the hearing->speaking mirror — strongly when it was pleasure,
   not at all when it was pain (``Brain._mirror_gate``): that is how being
   rewarded for repeating a word teaches it to repeat words.  The earliest
   recalled moment is also stored as a memory (replayed again in sleep).
3. **Pain makes it flinch**: it freezes for a moment (no new movement or
   sound) and is aroused (noradrenaline).  Strong pain wakes it up.
4. **The feeling lingers** for a second or two: it is part of the creature's
   state, shown in the dashboard and reported in its thought log
   ("ouch — that hurts").

This module holds the feelings themselves (their levels, the teaching
signal still to be delivered, counts); the recall is done by the brain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

KINDS = ("pleasure", "pain")
CREDITS = ("auto", "speech", "movement", "now")


@dataclass
class HedonicConfig:
    #: the teaching signal at intensity 1 (pain: punishment): a reward of
    #: ``onset`` at once, then ``rate`` per second while it lasts.  With the
    #: default dopamine system this holds the dopamine burst (or dip) near its
    #: maximum from an intensity of about 0.8, so above that, intensity mostly
    #: sets how long it lasts.
    onset: float = 0.4
    rate: float = 3.0
    min_s: float = 0.2  # how long the teaching signal lasts at a tiny intensity...
    per_intensity_s: float = 0.5  # ...plus this much at intensity 1
    pain_gain: float = 1.0  # pain's teaching signal relative to pleasure's
    linger_s: float = 1.5  # time constant of the felt level
    flinch_s: float = 0.4  # pain: no new movement or sound for this long
    arousal: float = 0.6  # pain: phasic noradrenaline at intensity 1
    wake: float = 0.6  # how strongly pain (x intensity) stimulates a sleeping brain; pleasure a third of it
    # recall (awake replay of what it just did)
    recall: bool = True
    recall_window_s: float = 8.0  # its last words are recalled if it said them this recently
    recall_gap_s: float = 1.6  # syllables closer together than this belong to one utterance
    recall_move_s: float = 3.0  # its last movement is recalled if it began this recently
    recall_ms: float = 80.0  # each recalled act is replayed this long
    recall_max: int = 6  # at most this many acts are recalled
    recall_wait_s: float = 0.4  # wait at most this long for an act in progress to finish
    max_intensity: float = 1.0


@dataclass
class Hedonic:
    """The felt levels of pain and pleasure, and the teaching signal still to come."""

    cfg: HedonicConfig = field(default_factory=HedonicConfig)
    pleasure: float = 0.0
    pain: float = 0.0
    # teaching signals in progress: [signed reward per second, seconds left, signed onset still to come]
    pulses: list[list[float]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"pleasure": 0, "pain": 0})
    totals: dict[str, float] = field(default_factory=lambda: {"pleasure": 0.0, "pain": 0.0})
    last: dict[str, Any] = field(default_factory=dict)

    def duration(self, intensity: float) -> float:
        return self.cfg.min_s + self.cfg.per_intensity_s * float(intensity)

    def add(self, kind: str, intensity: float, duration_s: float, t_s: float, source: str = "") -> None:
        """Start feeling ``kind`` (pleasure or pain) at ``intensity`` in [0, 1]."""
        if kind not in KINDS:
            raise ValueError(f"kind must be 'pleasure' or 'pain', not {kind!r}")
        x = float(np.clip(intensity, 0.0, self.cfg.max_intensity))
        sign = 1.0 if kind == "pleasure" else -self.cfg.pain_gain
        self.pulses.append([sign * self.cfg.rate * x, float(duration_s), sign * self.cfg.onset * x])
        setattr(self, kind, max(getattr(self, kind), x))
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.totals[kind] = self.totals.get(kind, 0.0) + x
        self.last = {"kind": kind, "intensity": round(x, 3), "t": round(float(t_s), 3), "source": source}

    def step(self, dt_s: float) -> float:
        """Advance; return the reward (+) or punishment (-) for this step."""
        r = 0.0
        keep = []
        for p in self.pulses:
            d = min(dt_s, p[1])
            r += p[0] * d + p[2]
            p[1] -= d
            p[2] = 0.0
            if p[1] > 1e-9:
                keep.append(p)
        self.pulses = keep
        f = float(np.exp(-dt_s / self.cfg.linger_s))
        self.pleasure *= f
        self.pain *= f
        return r

    @property
    def active(self) -> bool:
        return bool(self.pulses)

    def levels(self) -> dict[str, float]:
        return {"pleasure": float(self.pleasure), "pain": float(self.pain)}

    def state_dict(self) -> dict[str, Any]:
        return {"pleasure": self.pleasure, "pain": self.pain, "pulses": [list(p) for p in self.pulses],
                "counts": dict(self.counts), "totals": dict(self.totals), "last": dict(self.last),
                "cfg": asdict(self.cfg)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        if "cfg" in s:
            self.cfg = HedonicConfig(**s["cfg"])
        self.pleasure = float(s.get("pleasure", 0.0))
        self.pain = float(s.get("pain", 0.0))
        self.pulses = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in s.get("pulses", [])]
        self.counts.update({k: int(v) for k, v in s.get("counts", {}).items()})
        self.totals.update({k: float(v) for k, v in s.get("totals", {}).items()})
        self.last = dict(s.get("last", {}))


__all__ = ["Hedonic", "HedonicConfig", "KINDS", "CREDITS"]
