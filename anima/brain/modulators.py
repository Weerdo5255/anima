"""Neuromodulatory systems.

Four diffuse chemical systems broadcast global signals that change *how* the
whole network computes and learns, following Doya's (2002) proposal
("Metalearning and neuromodulation", Neural Networks 15):

dopamine (VTA/SNc)
    Phasic part = reward prediction error (Schultz, Dayan & Montague 1997);
    it is the third factor that turns eligibility traces into weight changes.
    Tonic part = recent average reward; sets motor "vigor" (Niv et al. 2007).
noradrenaline (locus coeruleus)
    Arousal and surprise.  Raises neural gain, sharpening competition
    (Aston-Jones & Cohen 2005).  Near zero in REM sleep.
acetylcholine (basal forebrain)
    High in waking and REM, low in slow-wave sleep.  Suppresses spike-
    frequency adaptation (so its absence lets cortex fall into slow up/down
    oscillations) and gates the learning rate of cortical plasticity
    (Hasselmo 2006).
serotonin (raphe)
    Slow mood / patience signal tracking the long-run balance of reward
    and punishment.  Low serotonin -> impulsive, quicker action release
    (Doya 2002; Miyazaki et al. 2014).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ModulatorConfig:
    tau_da: float = 0.2  # s, duration of a phasic dopamine transient
    tau_expect: float = 8.0  # s, reward expectation (baseline for RPE)
    tau_tonic: float = 30.0  # s, tonic dopamine (average reward)
    tau_mood: float = 120.0  # s, serotonin
    tau_ne: float = 1.5  # s, decay of phasic noradrenaline
    tau_ach: float = 3.0  # s, acetylcholine smoothing
    rpe_gain: float = 2.0  # dopamine impulse per unit of reward prediction error
    surprise_gain: float = 1.0


class Neuromodulators:
    """Scalar neuromodulator levels updated every brain tick."""

    ACH_TARGET = {"awake": 0.8, "nrem": 0.1, "rem": 0.9}
    NE_BASE = {"awake": 0.3, "nrem": 0.1, "rem": 0.0}

    def __init__(self, config: ModulatorConfig | None = None) -> None:
        self.cfg = config or ModulatorConfig()
        self.expected_rate = 0.0  # expected reward per second
        self.dopamine = 0.0  # phasic, signed, in [-1, 1]
        self.dopamine_tonic = 0.2  # in [0, 1]
        self.noradrenaline = 0.3
        self.acetylcholine = 0.8
        self.serotonin = 0.5
        self.last_rpe = 0.0
        self._ne_phasic = 0.0
        self._mood = 0.0

    def update(self, dt_s: float, reward: float, surprise: float,
               stage: str, attention: float = 0.0, rpe: float | None = None) -> None:
        """Advance all systems by ``dt_s`` seconds.

        reward: total reward delivered during this tick (positive = good,
            negative = punishment).
        surprise: sensory prediction error in [0, ~1].
        stage: "awake", "nrem" or "rem".
        attention: salience of current external input in [0, 1].
        rpe: reward prediction error from a critic (ventral striatum); if
            omitted, reward minus the running average reward rate is used.
        """
        c = self.cfg
        # --- dopamine: each reward prediction error adds an impulse that
        # decays over ~tau_da; the weight change it causes scales with the
        # size of the error (not clipped to a fixed pulse).
        if rpe is None:
            rpe = reward - self.expected_rate * dt_s
        self.last_rpe = rpe
        self.dopamine = float(np.clip(self.dopamine * np.exp(-dt_s / c.tau_da)
                                      + c.rpe_gain * rpe, -1.0, 1.0))
        self.expected_rate += (reward - self.expected_rate * dt_s) / c.tau_expect
        tonic_target = float(np.clip(0.2 + 2.0 * self.expected_rate, 0.0, 1.0))
        self.dopamine_tonic += (tonic_target - self.dopamine_tonic) * min(1.0, dt_s / c.tau_tonic)
        reward_rate = reward / dt_s
        # --- noradrenaline: baseline by stage + phasic surprise / salience
        self._ne_phasic *= np.exp(-dt_s / c.tau_ne)
        self._ne_phasic = max(self._ne_phasic, c.surprise_gain * surprise, attention)
        base = self.NE_BASE.get(stage, 0.3)
        gate = 0.0 if stage == "rem" else (0.5 if stage == "nrem" else 1.0)
        self.noradrenaline = float(np.clip(base + 0.7 * gate * self._ne_phasic, 0.0, 1.0))
        # --- acetylcholine: set by sleep stage, raised by attention when awake
        target = self.ACH_TARGET.get(stage, 0.8)
        if stage == "awake":
            target = min(1.0, target + 0.2 * attention)
        self.acetylcholine += (target - self.acetylcholine) * min(1.0, dt_s / c.tau_ach)
        # --- serotonin: slow mood from the balance of reward and punishment
        self._mood += (reward_rate * dt_s - self._mood * dt_s) / c.tau_mood
        sleep_factor = {"awake": 1.0, "nrem": 0.7, "rem": 0.2}.get(stage, 1.0)
        self.serotonin = float(np.clip(sleep_factor * (0.5 + 2.0 * self._mood), 0.0, 1.0))

    def arouse(self, level: float) -> None:
        """A phasic burst of noradrenaline (e.g. pain), at least ``level``."""
        self._ne_phasic = max(self._ne_phasic, float(level))

    def levels(self) -> dict[str, float]:
        return {
            "dopamine": self.dopamine,
            "dopamine_tonic": self.dopamine_tonic,
            "noradrenaline": self.noradrenaline,
            "acetylcholine": self.acetylcholine,
            "serotonin": self.serotonin,
        }

    def state_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "cfg"}
        d["cfg"] = asdict(self.cfg)
        return d

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k == "cfg":
                self.cfg = ModulatorConfig(**v)
            else:
                setattr(self, k, v)
