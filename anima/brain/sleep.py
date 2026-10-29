"""Sleep / wake regulation.

Implements Borbely's two-process model (1982): the homeostatic sleep
pressure S (from :class:`~anima.brain.drives.Drives`) is compared with
upper and lower thresholds that are modulated by the circadian rhythm C.
Sleep starts when S exceeds the upper threshold and ends when S falls below
the lower one.  Inside sleep, NREM and REM alternate in an ultradian cycle
(~90 min in humans, compressed by the same factor as the day).

What sleep *does* is implemented elsewhere and is real, not cosmetic:

* the thalamus closes the sensory gate and motor output is paralysed;
* acetylcholine falls in NREM, which increases spike-frequency adaptation
  in cortex and produces slow (<2 Hz) up/down oscillations;
* the hippocampus replays salient waking episodes into cortex and motor
  areas together with the dopamine they earned, consolidating them
  (Wilson & McNaughton 1994; Gomperts et al. 2015);
* plastic synapses are gently scaled down (synaptic homeostasis
  hypothesis, Tononi & Cirelli 2014).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SleepConfig:
    # Fitted so that with default drive settings sleep runs ~23:00 -> ~07:00
    upper_base: float = 0.70  # sleep onset threshold at circadian midpoint
    upper_amp: float = 0.60
    lower_base: float = 0.30  # wake threshold at circadian midpoint
    lower_amp: float = 0.40
    cycle_fraction: float = 1.0 / 16.0  # ultradian cycle as a fraction of the day
    rem_fraction: float = 0.25  # share of each cycle spent in REM
    arousal_threshold_nrem: float = 1.6
    arousal_threshold_rem: float = 1.0
    arousal_tau_s: float = 2.0
    min_wake_s: float = 20.0  # no sleep right after waking up
    min_sleep_s: float = 20.0


class SleepController:
    def __init__(self, config: SleepConfig | None = None) -> None:
        self.cfg = config or SleepConfig()
        self.stage = "awake"
        self.time_in_stage = 0.0
        self.time_asleep = 0.0
        self.arousal = 0.0
        self.transitions: list[tuple[float, str]] = []

    @property
    def awake(self) -> bool:
        return self.stage == "awake"

    def thresholds(self, circadian: float) -> tuple[float, float]:
        c = self.cfg
        x = circadian - 0.5
        return c.upper_base + c.upper_amp * x, c.lower_base + c.lower_amp * x

    def stimulate(self, amount: float) -> None:
        """External stimulation (sound, touch) that may wake a sleeper."""
        self.arousal += amount

    def update(self, dt_s: float, t_s: float, sleep_pressure: float, circadian: float,
               day_length_s: float) -> str | None:
        """Advance; return the new stage if a transition happened."""
        c = self.cfg
        self.time_in_stage += dt_s
        self.arousal *= max(0.0, 1.0 - dt_s / c.arousal_tau_s)
        upper, lower = self.thresholds(circadian)
        new = None
        if self.stage == "awake":
            if sleep_pressure > upper and self.time_in_stage > c.min_wake_s:
                new = "nrem"
                self.time_asleep = 0.0
        else:
            self.time_asleep += dt_s
            woken = self.arousal > (c.arousal_threshold_rem if self.stage == "rem"
                                    else c.arousal_threshold_nrem)
            if (sleep_pressure < lower and self.time_asleep > c.min_sleep_s) or woken:
                new = "awake"
            else:
                cycle = c.cycle_fraction * day_length_s
                pos = self.time_asleep % cycle
                in_rem = pos > (1.0 - c.rem_fraction) * cycle
                want = "rem" if in_rem else "nrem"
                if want != self.stage:
                    new = want
        if new is not None and new != self.stage:
            self.stage = new
            self.time_in_stage = 0.0
            self.arousal = 0.0
            self.transitions.append((t_s, new))
            if len(self.transitions) > 1000:
                self.transitions = self.transitions[-1000:]
            return new
        return None

    def force(self, stage: str, t_s: float) -> None:
        if stage not in ("awake", "nrem", "rem"):
            raise ValueError(stage)
        if stage != self.stage:
            self.stage = stage
            self.time_in_stage = 0.0
            if stage != "awake":
                self.time_asleep = 0.0
            self.transitions.append((t_s, stage))

    def state_dict(self) -> dict:
        return {"stage": self.stage, "time_in_stage": self.time_in_stage,
                "time_asleep": self.time_asleep, "arousal": self.arousal,
                "transitions": list(self.transitions), "cfg": asdict(self.cfg)}

    def load_state_dict(self, s: dict) -> None:
        self.cfg = SleepConfig(**s["cfg"])
        self.stage = str(s["stage"])
        self.time_in_stage = float(s["time_in_stage"])
        self.time_asleep = float(s["time_asleep"])
        self.arousal = float(s["arousal"])
        self.transitions = [(float(t), str(st)) for t, st in s["transitions"]]
