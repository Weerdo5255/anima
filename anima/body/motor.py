"""Turning motor-population activity into discrete movements.

The motor channels compete through shared inhibition (winner-take-all).
Whenever one channel's activity crosses threshold and clearly beats the
runner-up, the body commits to that movement for ``duration_ms``.  During
the movement the winning channel receives a commitment current and the
losers are suppressed — the thalamo-cortical loop that the basal ganglia
disinhibit for the selected action.  This also concentrates eligibility on
the synapses that actually produced the chosen action, which is what lets
dopamine assign credit correctly.

No input is required for any of this: background noise alone makes the
channels fluctuate, so a resting animal still occasionally twitches, turns
and wanders (exploration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..brain.brain import Brain


@dataclass
class MotorConfig:
    threshold: float = 0.28  # channel activity needed to release an action
    margin: float = 1.25  # winner must beat runner-up by this factor
    duration_ms: float = 200.0
    refractory_ms: float = 120.0  # pause after each movement so the competition resets
    feedback_on: float = 7.0
    feedback_off: float = -3.0


class MotorDecoder:
    def __init__(self, brain: "Brain", group: str = "motor", config: MotorConfig | None = None) -> None:
        self.brain = brain
        self.group = group
        self.cfg = config or MotorConfig()
        self.labels = brain.outputs[group].labels
        self.current: int = -1
        self.until = -1.0
        self.started = -1.0
        self.refractory_until = -1.0
        self.counts = np.zeros(len(self.labels), dtype=int)

    def update(self, t_ms: float) -> tuple[str, int] | None:
        """Return ("start", k), ("end", k) or None."""
        b, cfg = self.brain, self.cfg
        if self.current >= 0:
            if t_ms >= self.until:
                k = self.current
                self.current = -1
                self.refractory_until = t_ms + cfg.refractory_ms
                b.set_feedback(self.group, 0.0)
                b.end_action(self.group)
                return ("end", k)
            return None
        if not b.awake or t_ms < self.refractory_until or b.frozen:
            return None
        act = b.channel_activity(self.group)
        order = np.argsort(act)
        best, second = int(order[-1]), float(act[order[-2]]) if act.size > 1 else 0.0
        thr = cfg.threshold * b.action_threshold_scale()
        if act[best] < thr or act[best] < cfg.margin * second:
            return None
        self.current = best
        self.started = t_ms
        self.until = t_ms + cfg.duration_ms
        fb = np.full(len(self.labels), cfg.feedback_off)
        fb[best] = cfg.feedback_on
        b.set_feedback(self.group, fb)
        b.note_action(self.group, best)
        self.counts[best] += 1
        return ("start", best)

    @property
    def busy(self) -> bool:
        return self.current >= 0

    def state_dict(self) -> dict:
        return {"current": self.current, "until": self.until, "started": self.started,
                "refractory_until": self.refractory_until, "counts": self.counts.copy()}

    def load_state_dict(self, s: dict) -> None:
        self.current = int(s["current"])
        self.until = float(s["until"])
        self.started = float(s["started"])
        self.refractory_until = float(s["refractory_until"])
        self.counts[...] = s["counts"]
