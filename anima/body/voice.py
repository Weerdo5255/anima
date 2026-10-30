"""Hearing and speaking.

anima does not read text.  What you type is turned into *sound*: a timed
stream of phonemes played into the auditory population, one phoneme every
~60-100 ms, so listening takes time exactly as it does for a child.

It does not write text either.  It has a vocal tract with a small
articulatory repertoire (12 consonants x 5 vowels).  A syllable is produced
when the vocal *gate* population fires strongly enough; which consonant and
vowel come out is decided by two winner-take-all competitions.  The sound it
makes is fed back into its own ears (self-hearing), which is how it learns
the mapping between hearing a sound and making it, and therefore how it
becomes able to echo you.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..brain.brain import Brain

LETTER_MAP = {"c": "k", "q": "k", "x": "k", "j": "d", "f": "p", "v": "b", "z": "s",
              "r": "l", "y": "i"}
POSITIVE_SIGNALS = (":)", ":-)", ":d", "<3", "*smile*", "*hug*", "*pat*", "*laugh*")
NEGATIVE_SIGNALS = (":(", ":-(", "*frown*", "*sigh*")


def parse_social_signals(text: str) -> tuple[str, float]:
    """Strip emoticons / gestures; return (remaining text, valence)."""
    low = text.lower()
    valence = 0.0
    for sig in POSITIVE_SIGNALS:
        if sig in low:
            valence += 0.5 * low.count(sig)
            low = low.replace(sig, " ")
    for sig in NEGATIVE_SIGNALS:
        if sig in low:
            valence -= 0.5 * low.count(sig)
            low = low.replace(sig, " ")
    return low, float(np.clip(valence, -1.0, 1.0))


def syllabify(text: str, consonants: list[str] | None = None, vowels: list[str] | None = None) -> list[str]:
    """The syllables of ``text`` the way the creature can say them: each
    vowel with the consonant just before it ("mama" -> ["ma", "ma"], "ma ma"
    too; "hello" -> ["he", "lo"]).  Letters are mapped to its phoneme set
    as when it hears them; consonants with no vowel after them are dropped
    (it cannot say "dog", only "do")."""
    if consonants is None or vowels is None:
        from ..brain.presets import CONSONANTS, VOWELS
        consonants, vowels = consonants or CONSONANTS, vowels or VOWELS
    out: list[str] = []
    pending = ""
    for ch in text.lower():
        p = LETTER_MAP.get(ch, ch)
        if p in vowels:
            out.append(pending + p)
            pending = ""
        elif p in consonants:
            pending = p
        else:
            pending = ""
    return out


def text_to_phonemes(text: str, consonants: list[str], vowels: list[str],
                     cons_ms: float = 60.0, vowel_ms: float = 100.0, gap_ms: float = 15.0,
                     pause_ms: float = 150.0) -> list[tuple[str | None, float]]:
    """Map letters to the agent's phoneme inventory, with durations.

    ``None`` entries are silences.  Unknown characters are skipped.
    """
    out: list[tuple[str | None, float]] = []
    for ch in text.lower():
        p = LETTER_MAP.get(ch, ch)
        if p in consonants:
            out += [(p, cons_ms), (None, gap_ms)]
        elif p in vowels:
            out += [(p, vowel_ms), (None, gap_ms)]
        elif ch.isspace() or ch in ",.;:!?-":
            if out and out[-1][0] is None and out[-1][1] >= pause_ms:
                continue
            if out and out[-1][0] is None:
                out[-1] = (None, pause_ms)
            else:
                out.append((None, pause_ms))
    return out


@dataclass
class _Stream:
    events: deque = field(default_factory=deque)  # [channel or -1, remaining_ms]

    def push(self, seq: list[tuple[int, float]]) -> None:
        for ch, dur in seq:
            self.events.append([ch, float(dur)])

    def step(self, dt_ms: float) -> int:
        """Advance; return the channel sounding now (-1 = silence)."""
        while self.events and self.events[0][1] <= 0:
            self.events.popleft()
        if not self.events:
            return -1
        ev = self.events[0]
        ev[1] -= dt_ms
        return int(ev[0])

    @property
    def busy(self) -> bool:
        return any(e[0] >= 0 for e in self.events)

    def clear(self) -> None:
        self.events.clear()


class Ear:
    """Plays external speech and the agent's own voice into the audio port.

    Each heard phoneme also leaves a decaying trace (auditory sensory or
    "echoic" memory, ~1 s; Darwin, Turvey & Crowder 1972) that keeps its
    channel weakly active after the sound has stopped.  This is what lets a
    sound you just made still influence what the creature says next.  A new
    sound overwrites it: when the creature starts to speak, the echo of what
    it heard before is mostly masked (``self_masking``; the "suffix effect",
    Crowder & Morton 1969), so what it hears while it speaks is its own
    voice — which is what its hearing->speaking map learns from.
    """

    def __init__(self, labels: list[str], consonants: list[str], vowels: list[str],
                 echo_tau_s: float = 1.0, echo_gain: float = 0.5, self_masking: float = 0.3) -> None:
        self.labels = list(labels)
        self.index = {p: i for i, p in enumerate(self.labels)}
        self.consonants = consonants
        self.vowels = vowels
        self.external = _Stream()
        self.own = _Stream()
        self.current_external = -1
        self.current_own = -1
        self.echo_tau_s = echo_tau_s
        self.echo_gain = echo_gain
        self.self_masking = self_masking  # fraction of the echo left when it starts to speak
        self.echo = np.zeros(len(self.labels))

    def hear(self, text: str) -> int:
        """Queue external speech; returns the number of phonemes queued."""
        seq = text_to_phonemes(text, self.consonants, self.vowels)
        coded = [(self.index.get(p, -1) if p is not None else -1, d) for p, d in seq]
        self.external.push([(-1, 30.0)] + coded + [(-1, 200.0)])
        return sum(1 for p, _ in seq if p is not None)

    def hear_self(self, phonemes: list[str], durations: list[float], delay_ms: float = 20.0) -> None:
        seq = [(-1, delay_ms)] + [(self.index[p], d) for p, d in zip(phonemes, durations)]
        self.own.push(seq)
        self.echo *= self.self_masking

    def step(self, dt_ms: float) -> np.ndarray:
        self.current_external = self.external.step(dt_ms)
        self.current_own = self.own.step(dt_ms)
        if self.echo_tau_s > 0:
            self.echo *= np.exp(-dt_ms / 1000.0 / self.echo_tau_s)
        vec = np.zeros(len(self.labels))
        if self.current_external >= 0:
            vec[self.current_external] = 1.0
            self.echo[self.current_external] = 1.0
        if self.current_own >= 0:
            vec[self.current_own] = max(vec[self.current_own], 0.9)
            self.echo[self.current_own] = max(self.echo[self.current_own], 0.9)
        return np.maximum(vec, self.echo_gain * self.echo)

    def state_dict(self) -> dict:
        return {"external": [list(e) for e in self.external.events],
                "own": [list(e) for e in self.own.events],
                "current_external": self.current_external, "current_own": self.current_own,
                "echo": self.echo.copy()}

    def load_state_dict(self, s: dict) -> None:
        self.external.events = deque([int(c), float(d)] for c, d in s["external"])
        self.own.events = deque([int(c), float(d)] for c, d in s["own"])
        self.current_external = int(s["current_external"])
        self.current_own = int(s["current_own"])
        if "echo" in s:
            self.echo[...] = s["echo"]

    @property
    def hearing_external(self) -> bool:
        return self.current_external >= 0

    @property
    def external_pending(self) -> bool:
        return self.external.busy


@dataclass
class VoiceConfig:
    gate_threshold: float = 0.22  # vocal-gate activity needed to start a syllable
    consonant_threshold: float = 0.10  # below this the syllable is a bare vowel
    consonant_ms: float = 70.0
    vowel_ms: float = 150.0
    feedback_on: float = 7.0  # commitment current to the chosen articulators
    feedback_off: float = -2.5  # suppression of the competitors
    word_gap_ms: float = 450.0
    utterance_gap_ms: float = 1400.0
    max_word_syllables: int = 4
    # Breath: speech is produced on the out-breath, so bouts of babbling are
    # limited and must be followed by a pause to recover.
    breath_cost: float = 0.16  # per syllable
    breath_recovery_s: float = 4.0  # time constant of recovery while silent
    breath_inhibition: float = 8.0  # current that silences the gate when out of breath
    # Innate turn-taking: when someone stops speaking to it, the vocal gate is
    # briefly excited (infants vocalise in response to caregiver speech).  It
    # biases *when* it vocalises, never *what*.
    turn_boost: float = 3.0
    turn_window_ms: float = 600.0


class Voice:
    """Reads the vocal populations and turns their activity into syllables."""

    def __init__(self, brain: "Brain", ear: Ear | None, config: VoiceConfig | None = None,
                 gate: str = "vocal.gate", consonants: str = "vocal.C", vowels: str = "vocal.V") -> None:
        self.brain = brain
        self.ear = ear
        self.cfg = config or VoiceConfig()
        self.gate, self.cons, self.vows = gate, consonants, vowels
        self.cons_labels = brain.outputs[consonants].labels
        self.vow_labels = brain.outputs[vowels].labels
        self.articulating_until = -1.0
        self.word: list[str] = []
        self.utterance: list[str] = []
        self.last_syllable_t = -1e9
        self.syllables_total = 0
        self.breath = 1.0
        self._last_t = 0.0
        self._turn_until = -1.0
        self._was_hearing = False
        self.history: deque[tuple[float, str]] = deque(maxlen=2000)

    def update(self, t_ms: float) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        cfg = self.cfg
        b = self.brain
        dt_s = max(0.0, (t_ms - self._last_t) / 1000.0)
        self._last_t = t_ms
        if self.articulating_until < 0:
            self.breath += (1.0 - self.breath) * min(1.0, dt_s / cfg.breath_recovery_s)
        hearing = self.ear is not None and self.ear.external_pending
        if self._was_hearing and not hearing:
            self._turn_until = t_ms + cfg.turn_window_ms  # they stopped talking: my turn
        self._was_hearing = hearing
        turn = cfg.turn_boost if t_ms < self._turn_until else 0.0
        b.set_feedback(self.gate, turn - cfg.breath_inhibition * (1.0 - self.breath) ** 2)
        if self.articulating_until >= 0 and t_ms >= self.articulating_until:
            self.articulating_until = -1.0
            b.set_feedback(self.cons, 0.0)
            b.set_feedback(self.vows, 0.0)
            b.end_action(self.cons)
            b.end_action(self.vows)
            b.end_action(self.gate)
        # word / utterance boundaries from silence
        silence = t_ms - self.last_syllable_t
        if self.word and silence > cfg.word_gap_ms and self.articulating_until < 0:
            self.utterance.append("".join(self.word))
            self.word = []
        if self.utterance and silence > cfg.utterance_gap_ms:
            text = " ".join(self.utterance)
            self.utterance = []
            events.append(("utterance", {"text": text, "t": t_ms / 1000.0}))
        if self.articulating_until >= 0 or not b.awake or b.frozen:
            return events
        gate = float(b.channel_activity(self.gate)[0])
        if gate < cfg.gate_threshold * b.action_threshold_scale() or self.breath < cfg.breath_cost:
            return events
        c_act = b.channel_activity(self.cons)
        v_act = b.channel_activity(self.vows)
        v = int(np.argmax(v_act))
        c = int(np.argmax(c_act)) if c_act.max() > cfg.consonant_threshold else -1
        syl = (self.cons_labels[c] if c >= 0 else "") + self.vow_labels[v]
        dur = (cfg.consonant_ms if c >= 0 else 0.0) + cfg.vowel_ms
        self.articulating_until = t_ms + dur
        fb_c = np.full(len(self.cons_labels), cfg.feedback_off)
        if c >= 0:
            fb_c[c] = cfg.feedback_on
        fb_v = np.full(len(self.vow_labels), cfg.feedback_off)
        fb_v[v] = cfg.feedback_on
        b.set_feedback(self.cons, fb_c)
        b.set_feedback(self.vows, fb_v)
        b.note_action(self.gate, 0)
        b.note_action(self.cons, c)
        act = b.note_action(self.vows, v)
        if self.ear is not None:
            ph = ([self.cons_labels[c]] if c >= 0 else []) + [self.vow_labels[v]]
            du = ([cfg.consonant_ms] if c >= 0 else []) + [cfg.vowel_ms]
            self.ear.hear_self(ph, du)
        if len(self.word) >= cfg.max_word_syllables:
            self.utterance.append("".join(self.word))
            self.word = []
        self.word.append(syl)
        self.breath = max(0.0, self.breath - cfg.breath_cost)
        self.last_syllable_t = t_ms + dur
        self.syllables_total += 1
        self.history.append((t_ms / 1000.0, syl))
        events.append(("syllable", {"text": syl, "t": t_ms / 1000.0, "act": act}))
        return events

    @property
    def speaking(self) -> bool:
        return self.articulating_until >= 0

    def partial(self) -> str:
        return " ".join(self.utterance + (["".join(self.word)] if self.word else []))

    def state_dict(self) -> dict:
        return {"syllables_total": self.syllables_total,
                "history": [list(h) for h in self.history],
                "articulating_until": self.articulating_until, "word": list(self.word),
                "utterance": list(self.utterance), "last_syllable_t": self.last_syllable_t,
                "breath": self.breath, "last_t": self._last_t, "turn_until": self._turn_until,
                "was_hearing": self._was_hearing}

    def load_state_dict(self, s: dict) -> None:
        self.syllables_total = int(s["syllables_total"])
        self.history = deque(((float(t), str(x)) for t, x in s["history"]), maxlen=2000)
        if "breath" in s:
            self.articulating_until = float(s["articulating_until"])
            self.word = [str(x) for x in s["word"]]
            self.utterance = [str(x) for x in s["utterance"]]
            self.last_syllable_t = float(s["last_syllable_t"])
            self.breath = float(s["breath"])
            self._last_t = float(s["last_t"])
            self._turn_until = float(s.get("turn_until", -1.0))
            self._was_hearing = bool(s.get("was_hearing", False))
