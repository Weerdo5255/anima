"""Hypothalamic drives: the reason the brain does anything at all.

An LLM is a function that is evaluated when someone calls it.  An animal is a
set of homeostatic variables that drift away from their set points whether
or not anyone is around.  The growing error *is* motivation: it is fed to
the network as interoceptive input (so it changes ongoing activity and
which actions win), and its reduction is rewarding (Hull's drive-reduction
theory; Keramati & Gutkin 2014 "Homeostatic reinforcement learning").

Drives (all in [0, 1]):

hunger      1 - energy.  Energy is burnt by living and moving; food restores it.
fatigue     Sleep pressure, "process S" of Borbely's two-process model:
            rises while awake (faster when the brain is very active), decays
            during sleep.  Interacts with the circadian "process C".
boredom     Rises when the world is predictable, falls with novelty /
            prediction error (curiosity; Schmidhuber 1991, Oudeyer 2007).
loneliness  Rises without social contact, falls when someone talks to it
            (infants' babbling is shaped by social contingency;
            Goldstein et al. 2003).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

DRIVE_NAMES = ("hunger", "fatigue", "boredom", "loneliness")


@dataclass
class DriveConfig:
    day_length_s: float = 1200.0  # one circadian cycle (compressed so you can watch it)
    start_phase: float = 0.30  # fraction of the day at t=0 (0 = midnight, 0.25 = 6 am)
    boredom_rise_s: float = 60.0  # time constant of boredom build-up
    boredom_relief: float = 0.3  # how strongly novelty relieves boredom
    loneliness_rise_s: float = 90.0
    social_relief: float = 0.35  # loneliness removed by one contact
    activity_fatigue: float = 0.5  # extra sleep pressure from high neural activity


class Drives:
    def __init__(self, config: DriveConfig | None = None) -> None:
        self.cfg = config or DriveConfig()
        self.sleep_pressure = 0.15
        self.boredom = 0.3
        self.loneliness = 0.35
        self.energy = 0.75  # mirrored from the body's physiology each tick
        self.t_s = 0.0

    # --------------------------------------------------------------- clock
    @property
    def phase(self) -> float:
        """Time of day as a fraction in [0, 1)."""
        return (self.cfg.start_phase + self.t_s / self.cfg.day_length_s) % 1.0

    def circadian(self) -> float:
        """Process C in [0, 1]: minimum ~05:00, maximum ~17:00 (as body temperature)."""
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * (self.phase - 0.2))

    def clock_string(self) -> str:
        minutes = int(self.phase * 24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    # --------------------------------------------------------------- update
    def update(self, dt_s: float, awake: bool, energy: float,
               novelty: float, cortical_rate_hz: float) -> None:
        c = self.cfg
        self.t_s += dt_s
        self.energy = float(np.clip(energy, 0.0, 1.0))
        # Process S: saturating rise when awake, exponential decay asleep.
        day = c.day_length_s
        if awake:
            tau_w = 0.7 * day / (1.0 + c.activity_fatigue * max(0.0, cortical_rate_hz - 4.0) / 4.0)
            self.sleep_pressure += (1.0 - self.sleep_pressure) * dt_s / tau_w
        else:
            tau_s = 0.2 * day
            self.sleep_pressure -= self.sleep_pressure * dt_s / tau_s
        # Boredom: drifts up, novelty pushes it down.  Not felt while asleep.
        if awake:
            self.boredom += (1.0 - self.boredom) * dt_s / c.boredom_rise_s
            self.boredom -= c.boredom_relief * novelty * self.boredom * dt_s
        self.loneliness += (1.0 - self.loneliness) * dt_s / c.loneliness_rise_s * (1.0 if awake else 0.2)
        self.boredom = float(np.clip(self.boredom, 0.0, 1.0))
        self.loneliness = float(np.clip(self.loneliness, 0.0, 1.0))
        self.sleep_pressure = float(np.clip(self.sleep_pressure, 0.0, 1.0))

    def social_contact(self, strength: float = 1.0) -> float:
        """Someone interacted with us.  Returns the loneliness relieved."""
        relief = self.cfg.social_relief * strength * self.loneliness
        self.loneliness = float(max(0.0, self.loneliness - relief))
        return relief

    # ---------------------------------------------------------------- views
    @property
    def hunger(self) -> float:
        return 1.0 - self.energy

    @property
    def fatigue(self) -> float:
        return self.sleep_pressure

    def vector(self) -> np.ndarray:
        return np.array([self.hunger, self.fatigue, self.boredom, self.loneliness])

    def as_dict(self) -> dict[str, float]:
        return dict(zip(DRIVE_NAMES, self.vector().tolist()))

    def state_dict(self) -> dict:
        return {"sleep_pressure": self.sleep_pressure, "boredom": self.boredom,
                "loneliness": self.loneliness, "energy": self.energy, "t_s": self.t_s,
                "cfg": asdict(self.cfg)}

    def load_state_dict(self, s: dict) -> None:
        self.cfg = DriveConfig(**s["cfg"])
        for k in ("sleep_pressure", "boredom", "loneliness", "energy", "t_s"):
            setattr(self, k, float(s[k]))
