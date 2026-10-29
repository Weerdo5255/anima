"""Passive skill measurement.

A teacher needs to know how the pupil is doing without stopping it to test
it.  :class:`SkillMeter` listens to the creature's own events (syllables,
meals, what it hears) on the creature's thread and keeps rolling windows
from which progress is computed on demand:

food per minute
    Meals in the last five minutes (foraging *skill* is tested by an exam,
    see :mod:`.exams`: with several pieces of food in the arena, natural
    moves rarely show whether it can follow its nose).
target share
    Fraction of recent syllables starting with the target sound (e.g. "m").
echo rate
    After the teacher says a word, does the creature's first syllable within
    3 s start with the consonant it just heard?  Compared with chance (how
    often it uses that consonant anyway).
asking when hungry
    Share of target syllables produced while hungry vs. while sated — the
    signature of a sound used *to ask for* food rather than at random.
energy upkeep
    Mean and minimum energy over the last minutes (independent survival).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..body.voice import LETTER_MAP
from ..brain.presets import CONSONANTS

HUNGRY = 0.45  # hunger at or above which a call counts as "hungry"
SATED = 0.30  # hunger below which it counts as "sated"


def first_consonant(text: str) -> str | None:
    """The consonant a word starts with, in the creature's phoneme set."""
    for ch in text.lower():
        p = LETTER_MAP.get(ch, ch)
        if p in CONSONANTS:
            return p
        if p.isalpha():
            return None  # starts with a vowel
    return None


@dataclass
class SkillReport:
    food_per_min: float = 0.0
    target_share: float = float("nan")
    target_n: int = 0
    echo_rate: float = float("nan")
    echo_chance: float = float("nan")
    echo_n: int = 0
    echo_unanswered: int = 0
    ask_hungry: float = float("nan")
    ask_full: float = float("nan")
    ask_n_hungry: int = 0
    ask_n_full: int = 0
    energy_mean: float = float("nan")
    energy_min: float = float("nan")
    energy_minutes: float = 0.0

    def compact(self) -> dict[str, Any]:
        """Rounded, NaN-free dict for prompts and logs."""
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, float):
                out[k] = None if math.isnan(v) else round(v, 3)
            else:
                out[k] = v
        return out


class SkillMeter:
    def __init__(self, agent: Any, target: str = "m", syllable_window: int = 100, ask_window: int = 60,
                 energy_minutes: float = 10.0) -> None:
        self.agent = agent
        self.target = target
        self._lock = threading.Lock()
        self.meals: deque[float] = deque(maxlen=500)
        self.syllables: deque[tuple[float, str, float]] = deque(maxlen=400)
        self.recent_target: deque[bool] = deque(maxlen=syllable_window)
        self.ask_hungry: deque[bool] = deque(maxlen=ask_window)
        self.ask_full: deque[bool] = deque(maxlen=ask_window)
        self.echo_trials: deque[tuple[str, str | None]] = deque(maxlen=60)
        self.energy: deque[float] = deque(maxlen=int(energy_minutes * 60))
        self._pending_echo: dict[str, Any] | None = None
        self._unanswered = 0
        self._was_hearing = False
        self._last_energy_t = -1.0
        agent.on("ate", self._on_ate)
        agent.on("syllable", self._on_syllable)
        agent.add_tick_hook(self._on_tick)

    def detach(self) -> None:
        a = self.agent
        a.off("ate", self._on_ate)
        a.off("syllable", self._on_syllable)
        a.remove_tick_hook(self._on_tick)

    # ------------------------------------------------------------ listeners (agent thread)
    def _on_ate(self, e: dict[str, Any]) -> None:
        with self._lock:
            self.meals.append(e["t"])

    def _on_syllable(self, e: dict[str, Any]) -> None:
        text = e["text"]
        hunger = float(self.agent.brain.drives.hunger)
        is_target = text.startswith(self.target)
        with self._lock:
            self.syllables.append((e["t"], text, hunger))
            self.recent_target.append(is_target)
            if hunger >= HUNGRY:
                self.ask_hungry.append(is_target)
            elif hunger < SATED:
                self.ask_full.append(is_target)
            p = self._pending_echo
            if p is not None and p.get("end") is not None and e["t"] >= p["end"]:
                cons = text[0] if text[0] in CONSONANTS else None
                self.echo_trials.append((p["target"], cons))
                self._pending_echo = None

    def _on_tick(self, agent: Any) -> None:
        t = agent.t_ms / 1000.0
        if t - self._last_energy_t >= 1.0:
            self._last_energy_t = t
            if agent.brain.awake:
                with self._lock:
                    self.energy.append(float(agent.body.energy))
        ear = agent.body.ear
        hearing = bool(ear is not None and ear.external_pending)
        with self._lock:
            p = self._pending_echo
            if p is not None:
                if p.get("end") is None and self._was_hearing and not hearing:
                    p["end"] = t  # the teacher just finished speaking
                elif p.get("end") is not None and t - p["end"] > 3.0:
                    self._pending_echo = None
                    self._unanswered += 1
        self._was_hearing = hearing

    # ------------------------------------------------------------ teacher hooks
    def note_teacher_said(self, text: str) -> None:
        """Called when the teacher speaks: opens an echo trial."""
        c = first_consonant(text)
        with self._lock:
            self._pending_echo = None if c is None else {"target": c, "end": None}

    def clear(self, *what: str) -> None:
        """Start measuring afresh: "target", "echo", "ask" and/or "energy"."""
        with self._lock:
            if "target" in what:
                self.recent_target.clear()
            if "echo" in what:
                self.echo_trials.clear()
                self._pending_echo = None
                self._unanswered = 0
            if "ask" in what:
                self.ask_hungry.clear()
                self.ask_full.clear()
            if "energy" in what:
                self.energy.clear()

    def set_target(self, target: str) -> None:
        with self._lock:
            if target != self.target:
                self.target = target
                self.recent_target.clear()
                self.ask_hungry.clear()
                self.ask_full.clear()

    # ------------------------------------------------------------ report
    def report(self) -> SkillReport:
        now = self.agent.t_ms / 1000.0
        with self._lock:
            r = SkillReport()
            recent_meals = [t for t in self.meals if now - t <= 300.0]
            r.food_per_min = len(recent_meals) / min(5.0, max(now / 60.0, 1e-3))
            if self.recent_target:
                r.target_n = len(self.recent_target)
                r.target_share = float(np.mean(self.recent_target))
            trials = list(self.echo_trials)
            r.echo_unanswered = self._unanswered
            if trials:
                r.echo_n = len(trials)
                r.echo_rate = float(np.mean([c == tgt for tgt, c in trials]))
                cons = [s[0] for _, s, _ in self.syllables if s[0] in CONSONANTS]
                counts = {c: cons.count(c) / max(1, len(self.syllables)) for c in set(cons)}
                r.echo_chance = float(np.mean([counts.get(tgt, 0.0) for tgt, _ in trials]))
            if self.ask_hungry:
                r.ask_n_hungry = len(self.ask_hungry)
                r.ask_hungry = float(np.mean(self.ask_hungry))
            if self.ask_full:
                r.ask_n_full = len(self.ask_full)
                r.ask_full = float(np.mean(self.ask_full))
            if self.energy:
                r.energy_mean = float(np.mean(self.energy))
                r.energy_min = float(np.min(self.energy))
                r.energy_minutes = len(self.energy) / 60.0
            return r

    def recent_syllables(self, seconds: float = 30.0) -> list[str]:
        now = self.agent.t_ms / 1000.0
        with self._lock:
            return [s for t, s, _ in self.syllables if now - t <= seconds]

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"target": self.target,
                    "meals": list(self.meals),
                    "syllables": [list(x) for x in self.syllables],
                    "recent_target": list(self.recent_target),
                    "ask_hungry": list(self.ask_hungry), "ask_full": list(self.ask_full),
                    "echo_trials": [list(x) for x in self.echo_trials],
                    "energy": list(self.energy),
                    "unanswered": self._unanswered}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        with self._lock:
            self.target = s.get("target", self.target)
            self.meals.extend(float(t) for t in s.get("meals", []))
            self.syllables.extend((float(t), str(x), float(h)) for t, x, h in s.get("syllables", []))
            self.recent_target.extend(bool(x) for x in s.get("recent_target", []))
            self.ask_hungry.extend(bool(x) for x in s.get("ask_hungry", []))
            self.ask_full.extend(bool(x) for x in s.get("ask_full", []))
            self.echo_trials.extend((str(a), None if b is None else str(b))
                                    for a, b in s.get("echo_trials", []))
            self.energy.extend(float(x) for x in s.get("energy", []))
            self._unanswered = int(s.get("unanswered", 0))
