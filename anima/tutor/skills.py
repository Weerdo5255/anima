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
copying words
    After the teacher says a word ("ma ma", "bo bo"), how much of it does
    its reply repeat: the first consonant, the first syllable (consonant and
    vowel), or the whole word?  Each compared with chance (how often its
    replies would match by luck, from how often it says those syllables).
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

from ..body.voice import LETTER_MAP, parse_social_signals, syllabify
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


#: how much of the teacher's word a reply repeats, from least to most
LEVELS = ("none", "vowel", "consonant", "syllable", "word")
WORD_GAP_S = 0.45  # silence that ends its reply (as the voice ends a word)
REPLY_WINDOW_S = 3.0  # it must start replying within this long


def target_syllables(text: str) -> list[str]:
    """The syllables of what the teacher said (smiles and frowns left out)."""
    return syllabify(parse_social_signals(text)[0])


def _cons(syl: str) -> str:
    return syl[:-1]  # "" for a bare vowel


def score_reply(target: list[str], reply: list[str]) -> dict[str, Any]:
    """How much of ``target`` (the teacher's syllables) ``reply`` (its
    syllables) repeats.  ``level`` is one of :data:`LEVELS`: the whole word
    (all syllables in order, at the start of the reply), its first syllable,
    the first consonant, only the first vowel, or nothing."""
    if not target or not reply:
        return {"answered": bool(reply), "level": "none", "consonant": False, "vowel": False,
                "syllable": False, "word": False}
    t0, r0 = target[0], reply[0]
    cons = _cons(t0) != "" and _cons(r0) == _cons(t0)
    vowel = r0[-1] == t0[-1]
    syl = r0 == t0
    word = len(reply) >= len(target) and reply[:len(target)] == target
    level = "word" if word else "syllable" if syl else "consonant" if cons else "vowel" if vowel else "none"
    return {"answered": True, "level": level, "consonant": cons, "vowel": vowel, "syllable": syl, "word": word}


class WordTrial:
    """One "I say a word, you answer" exchange, followed on the creature's thread.

    Opened when the teacher speaks; its reply is the syllables it says after
    the teacher has finished, up to as many as the word has, or until it
    pauses (``WORD_GAP_S``).  ``closed`` once the reply is complete or it
    did not answer within ``REPLY_WINDOW_S``.
    """

    def __init__(self, said: str, t: float) -> None:
        self.said = said
        self.target = target_syllables(said)
        self.t = t
        self.end: float | None = None  # when the teacher finished speaking
        self.reply: list[str] = []
        self.acts: list[int] = []  # the brain's ids of the reply's syllables
        self.last_t = -1e9
        self.closed = False
        self.result: dict[str, Any] | None = None
        self.rewarded = ""  # what the teacher's reflexes gave for the reply

    def heard_end(self, t: float) -> None:
        if self.end is None:
            self.end = t

    def syllable(self, t: float, text: str, act: int | None = None) -> bool:
        """Its syllable at ``t``; returns True if this closed the trial."""
        if self.closed or self.end is None or t < self.end:
            return False
        self.reply.append(text)
        if act is not None:
            self.acts.append(int(act))
        self.last_t = t
        if len(self.reply) >= max(1, len(self.target)):
            return self._close()
        return False

    def tick(self, t: float, syllable_s: float = 0.22) -> bool:
        """Time passes; returns True if this closed the trial."""
        if self.closed or self.end is None:
            return False
        if self.reply and t - (self.last_t + syllable_s) > WORD_GAP_S:
            return self._close()
        if not self.reply and t - self.end > REPLY_WINDOW_S:
            return self._close()
        return False

    def _close(self) -> bool:
        self.closed = True
        self.result = {"said": self.said, "target": self.target, "reply": list(self.reply),
                       **score_reply(self.target, self.reply)}
        return True


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
    # copying the teacher's words (first consonant / first syllable / whole word), with chance levels
    copy_n: int = 0
    copy_consonant: float = float("nan")
    copy_syllable: float = float("nan")
    copy_word: float = float("nan")
    copy_syllable_chance: float = float("nan")
    copy_word_chance: float = float("nan")

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
        self.word_trials: deque[dict[str, Any]] = deque(maxlen=60)
        self._word: WordTrial | None = None
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
            w = self._word
            if w is not None and w.syllable(e["t"], text, e.get("act")):
                self._finish_word(w)

    def _finish_word(self, w: WordTrial) -> None:
        if w.result is not None and w.result["answered"] and w.target:
            self.word_trials.append({k: w.result[k] for k in ("said", "target", "reply", "level")})
        self._word = None

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
            w = self._word
            if w is not None:
                if self._was_hearing and not hearing:
                    w.heard_end(t)
                if w.tick(t):
                    self._finish_word(w)
        self._was_hearing = hearing

    # ------------------------------------------------------------ teacher hooks
    def note_teacher_said(self, text: str) -> None:
        """Called when the teacher speaks: opens an echo trial."""
        c = first_consonant(text)
        with self._lock:
            self._pending_echo = None if c is None else {"target": c, "end": None}
            w = WordTrial(text, self.agent.t_ms / 1000.0)
            self._word = w if w.target else None

    def clear(self, *what: str) -> None:
        """Start measuring afresh: "target", "echo", "ask" and/or "energy"."""
        with self._lock:
            if "target" in what:
                self.recent_target.clear()
            if "echo" in what:
                self.echo_trials.clear()
                self._pending_echo = None
                self._unanswered = 0
                self.word_trials.clear()
                self._word = None
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
            words = list(self.word_trials)
            if words:
                r.copy_n = len(words)
                rank = {lev: i for i, lev in enumerate(LEVELS)}
                lv = [rank[w["level"]] for w in words]
                r.copy_consonant = float(np.mean([x >= rank["consonant"] for x in lv]))
                r.copy_syllable = float(np.mean([x >= rank["syllable"] for x in lv]))
                r.copy_word = float(np.mean([x >= rank["word"] for x in lv]))
                said = [x for _, x, _ in self.syllables]
                freq = {x: said.count(x) / len(said) for x in set(said)} if said else {}
                r.copy_syllable_chance = float(np.mean([freq.get(w["target"][0], 0.0) for w in words]))
                r.copy_word_chance = float(np.mean([float(np.prod([freq.get(x, 0.0) for x in w["target"]]))
                                                    for w in words]))
            return r

    def recent_words(self, n: int = 5) -> list[dict[str, Any]]:
        """The last word trials: what the teacher said, what it replied, how well it matched."""
        with self._lock:
            return [dict(w) for w in list(self.word_trials)[-n:]]

    def last_words(self, gap_s: float = 1.6) -> tuple[str, float] | None:
        """What it said last — syllables closer together than ``gap_s`` —
        and how many seconds ago it started saying it."""
        now = self.agent.t_ms / 1000.0
        with self._lock:
            syl = [(t, x) for t, x, _ in self.syllables]
        if not syl:
            return None
        out = [syl[-1]]
        for t, x in reversed(syl[:-1]):
            if out[0][0] - t > gap_s or len(out) >= 6:
                break
            out.insert(0, (t, x))
        return " ".join(x for _, x in out), now - out[0][0]

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
                    "unanswered": self._unanswered,
                    "word_trials": [dict(w) for w in self.word_trials]}

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
            self.word_trials.extend(dict(w) for w in s.get("word_trials", []))
