"""What a teacher can do to the creature — deliberately, and by reflex.

Reinforcement only works if it is *contingent*: the smile has to come within
about a second of the behaviour it rewards (that is the window of the
creature's eligibility traces).  An LLM takes seconds to think, so teaching
is split in two, the way a human teacher's is:

* **deliberate actions** chosen by the teacher every few seconds: say a word,
  smile, frown, put food somewhere, and switch reflex rules on or off;
* **reflex rules** that run on the creature's own clock (every 10 ms tick)
  and respond within half a second: smile after the target syllable, feed it
  when it calls while hungry, keep food within reach while it is learning to
  forage, praise it when it copies a word.

Everything here is thread-safe: actions are posted to the creature's inbox
and executed on its own thread, so the creature never waits for the teacher.
"""

from __future__ import annotations

import heapq
import re
import threading
from collections import deque
from typing import Any, Callable

import numpy as np

from ..world.foraging import OFFER_PLACES
from .skills import HUNGRY, SATED, SkillMeter, first_consonant

WHERE = tuple(OFFER_PLACES)  # ahead, left, right, near
_ALLOWED_TEXT = re.compile(r"[^a-zA-Z :()*<3!?.,'-]")


def clean_text(text: str, limit: int = 40) -> str:
    """Sanitise something the creature will *hear*: letters, spaces, emoticons."""
    text = re.sub(r"<(?!3)", "", _ALLOWED_TEXT.sub("", str(text)))  # '<' only as in '<3'
    return text[:limit].strip()


def clean_note(text: str, limit: int = 240) -> str:
    """Sanitise free text that is only logged (thoughts, notes)."""
    return "".join(ch for ch in str(text) if ch.isprintable())[:limit].strip()


def clean_prefix(prefix: str, default: str = "m") -> str:
    p = re.sub(r"[^a-z]", "", str(prefix).lower())[:3]
    return p or default


# ======================================================================= rules
class Rule:
    """A reflex: reacts to the creature's events on its own thread."""

    name = "rule"

    def __init__(self, kit: "TeachingToolkit", **params: Any) -> None:
        self.kit = kit
        self.params = params
        self.last = -1e9
        self.fired = 0

    def on_syllable(self, t: float, text: str) -> None:
        pass

    def on_tick(self, t: float) -> None:
        pass

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class RewardSyllable(Rule):
    """Smile shortly after every syllable starting with ``prefix``."""

    name = "reward_syllable"

    def __init__(self, kit, prefix: str = "m", delay: float = 0.5, refractory: float = 2.0) -> None:
        super().__init__(kit, prefix=clean_prefix(prefix))
        self.delay, self.refractory = delay, refractory

    def on_syllable(self, t, text):
        if text.startswith(self.params["prefix"]) and t - self.last >= self.refractory:
            self.last = t
            self.fired += 1
            self.kit.schedule(t + self.delay, lambda: self.kit._hear(":)"), "praise (reflex)")


class FeedOnCall(Rule):
    """When it says the call syllable while hungry, put food right in front of it (and, if it calls when full, frown)."""

    name = "feed_on_call"

    def __init__(self, kit, prefix: str = "m", min_hunger: float = HUNGRY, delay: float = 0.3,
                 refractory: float = 6.0, frown_when_full: bool = True, full_below: float = SATED,
                 praise: bool = True, frown: str = ":(") -> None:
        super().__init__(kit, prefix=clean_prefix(prefix), min_hunger=min_hunger,
                         frown_when_full=bool(frown_when_full), praise=bool(praise))
        self.delay, self.refractory, self.full_below = delay, refractory, full_below
        self.frown_text = frown if frown in (":(", ":( :(") else ":("
        self.last_frown = -1e9
        self.frowned = 0

    def on_syllable(self, t, text):
        a = self.kit.agent
        if not (text.startswith(self.params["prefix"]) and a.brain.awake):
            return
        hunger = a.brain.drives.hunger
        if hunger >= self.params["min_hunger"] and t - self.last >= self.refractory:
            self.last = t
            self.fired += 1
            self.kit.schedule(t + self.delay, lambda: self.kit._place_food("ahead"), "feed (answering its call)")
            if self.params["praise"]:
                self.kit.schedule(t + self.delay, lambda: self.kit._hear(":)"), "praise (reflex)")
        elif (self.params["frown_when_full"] and hunger < self.full_below
              and t - self.last_frown >= self.refractory):
            # "you've just eaten": the call is for when it is hungry
            self.last_frown = t
            self.frowned += 1
            self.kit.schedule(t + self.delay, lambda: self.kit._hear(self.frown_text), "frown (it called while full)")


class GuideFood(Rule):
    """Scaffolding: while it is hungry and no food is near, put some within reach."""

    name = "guide_food"

    def __init__(self, kit, min_hunger: float = 0.35, near: float = 6.0, every: float = 1.0,
                 refractory: float = 8.0) -> None:
        super().__init__(kit, min_hunger=min_hunger)
        self.near, self.every, self.refractory = near, every, refractory
        self._next_check = 0.0

    def on_tick(self, t):
        if t < self._next_check:
            return
        self._next_check = t + self.every
        a = self.kit.agent
        w = a.world
        if not hasattr(w, "nearest_food_distance") or not a.brain.awake:
            return
        if (a.brain.drives.hunger >= self.params["min_hunger"] and t - self.last >= self.refractory
                and w.nearest_food_distance() > self.near):
            self.last = t
            self.fired += 1
            self.kit._place_food("near")
            self.kit._log(t, "guide: food within reach")


class EchoBack(Rule):
    """Answer each of its utterances by repeating its first syllable (turn-taking)."""

    name = "echo_back"

    def __init__(self, kit, delay: float = 0.8, refractory: float = 5.0) -> None:
        super().__init__(kit)
        self.delay, self.refractory = delay, refractory

    def on_utterance(self, t: float, text: str) -> None:
        if t - self.last >= self.refractory and text:
            self.last = t
            self.fired += 1
            word = text.split()[0]
            syl = word[:2] if len(word) >= 2 else word
            self.kit.schedule(t + self.delay, lambda: self.kit._say(f"{syl} {syl}"), f"say '{syl} {syl}' (echo)")


class RewardEcho(Rule):
    """After the teacher says a word, smile if its next syllable copies the consonant."""

    name = "reward_echo"

    def __init__(self, kit, window: float = 3.0) -> None:
        super().__init__(kit)
        self.window = window

    def on_syllable(self, t, text):
        trial = self.kit._echo_trial
        if trial is None or trial.get("end") is None or t < trial["end"]:
            return
        self.kit._echo_trial = None
        if t - trial["end"] <= self.window and text.startswith(trial["target"]):
            self.fired += 1
            self.kit.schedule(t + 0.2, lambda: self.kit._hear(":)"), "praise (copied me)")


RULES: dict[str, type[Rule]] = {r.name: r for r in
                                (RewardSyllable, FeedOnCall, GuideFood, EchoBack, RewardEcho)}


# ======================================================================= toolkit
class TeachingToolkit:
    def __init__(self, agent: Any, meter: SkillMeter, rng_seed: int = 0) -> None:
        self.agent = agent
        self.meter = meter
        self.rng = np.random.default_rng(rng_seed)
        self._rules: dict[str, Rule] = {}
        self._lock = threading.Lock()
        self._pending: list[tuple[float, int, Callable[[], None], str]] = []
        self._seq = 0
        self._echo_trial: dict[str, Any] | None = None
        self._was_hearing = False
        self.journal: deque[tuple[float, str]] = deque(maxlen=200)
        self.food_given = 0
        self.last_food_t = -1e9
        self.energy_cap: float | None = None
        self.wild_food = True
        agent.on("syllable", self._on_syllable)
        agent.on("utterance", self._on_utterance)
        agent.add_tick_hook(self._on_tick)

    def detach(self) -> None:
        self.agent.off("syllable", self._on_syllable)
        self.agent.off("utterance", self._on_utterance)
        self.agent.remove_tick_hook(self._on_tick)

    # ------------------------------------------------------------ deliberate actions (any thread)
    def say(self, text: str) -> str:
        text = clean_text(text)
        if text:
            self.agent.call_soon(lambda ag: self._say(text))
        return text

    def praise(self) -> None:
        self.agent.hear(":)")

    def scold(self) -> None:
        self.agent.hear(":(")

    def give_food(self, where: str = "ahead", min_interval_s: float = 4.0) -> bool:
        """Place food relative to the creature; rate-limited so food stays meaningful."""
        t = self.agent.t_ms / 1000.0
        if where not in WHERE or t - self.last_food_t < min_interval_s:
            return False
        self.last_food_t = t
        self.agent.call_soon(lambda ag: self._place_food(where))
        return True

    def set_rule(self, name: str, **params: Any) -> str:
        if name not in RULES:
            raise KeyError(f"unknown rule {name!r}; choose from {sorted(RULES)}")
        rule = RULES[name](self, **params)
        with self._lock:
            self._rules[name] = rule
        return rule.describe()

    def clear_rule(self, name: str) -> None:
        with self._lock:
            self._rules.pop(name, None)

    def clear_rules(self) -> None:
        with self._lock:
            self._rules.clear()

    def active_rules(self) -> list[str]:
        with self._lock:
            return [r.describe() for r in self._rules.values()]

    def rule_names(self) -> list[str]:
        with self._lock:
            return list(self._rules)

    def note(self, text: str) -> None:
        self._log(self.agent.t_ms / 1000.0, "note: " + clean_note(text, 160))

    def set_conditions(self, energy_cap: float | None = None, wild_food: bool = True) -> None:
        """Training conditions the tutor maintains (set per lesson, not by the teacher).

        energy_cap: keep the creature keen — mildly food-restricted, as animal
            trainers do — by holding its energy at or below this value, so that
            food stays motivating and the smell of food stays sharp.
        wild_food: whether natural food grows in the arena (False: the only
            food is what the teacher gives).
        """
        self.energy_cap = None if energy_cap is None else float(energy_cap)
        self.wild_food = bool(wild_food)
        self.agent.call_soon(lambda ag: self._apply_world())

    def conditions(self) -> dict[str, Any]:
        return {"energy_cap": self.energy_cap, "wild_food": self.wild_food}

    def _apply_world(self) -> None:
        w = self.agent.world
        if hasattr(w, "set_wild"):
            w.set_wild(self.wild_food)

    # ------------------------------------------------------------ agent-thread internals
    def _log(self, t: float, what: str) -> None:
        self.journal.append((t, what))

    def _hear(self, text: str) -> None:
        self.agent.hear(text)

    def _say(self, text: str) -> None:
        self.agent.hear(text)
        self.meter.note_teacher_said(text)
        c = first_consonant(text)
        self._echo_trial = None if c is None else {"target": c, "end": None}
        self._log(self.agent.t_ms / 1000.0, f"said '{text}'")

    def _place_food(self, where: str) -> None:
        w = self.agent.world
        if not hasattr(w, "offer_food"):
            return
        w.offer_food(where if where in OFFER_PLACES else "ahead", rng=self.rng)
        self.food_given += 1
        self._log(self.agent.t_ms / 1000.0, f"food placed {where}")

    def schedule(self, due_t: float, fn: Callable[[], None], what: str) -> None:
        with self._lock:
            self._seq += 1
            heapq.heappush(self._pending, (due_t, self._seq, fn, what))

    def _on_syllable(self, e: dict[str, Any]) -> None:
        with self._lock:
            rules = list(self._rules.values())
        for r in rules:
            r.on_syllable(e["t"], e["text"])

    def _on_utterance(self, e: dict[str, Any]) -> None:
        with self._lock:
            rules = list(self._rules.values())
        for r in rules:
            if hasattr(r, "on_utterance"):
                r.on_utterance(e["t"], e["text"])

    def _on_tick(self, agent: Any) -> None:
        t = agent.t_ms / 1000.0
        cap = self.energy_cap
        if cap is not None and agent.body.energy > cap:
            agent.body.energy = cap
        ear = agent.body.ear
        hearing = bool(ear is not None and ear.external_pending)
        trial = self._echo_trial
        if trial is not None and trial.get("end") is None and self._was_hearing and not hearing:
            trial["end"] = t
        self._was_hearing = hearing
        due = []
        with self._lock:
            while self._pending and self._pending[0][0] <= t:
                due.append(heapq.heappop(self._pending))
            rules = list(self._rules.values())
        for _, _, fn, what in due:
            fn()
            self._log(t, what)
        for r in rules:
            r.on_tick(t)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            rules = [{"name": r.name, "params": dict(r.params)} for r in self._rules.values()]
        return {"rules": rules, "journal": [list(x) for x in self.journal],
                "food_given": self.food_given, "conditions": self.conditions()}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.clear_rules()
        for r in s.get("rules", []):
            if r["name"] in RULES:
                self.set_rule(r["name"], **r.get("params", {}))
        self.journal.extend((float(t), str(x)) for t, x in s.get("journal", []))
        self.food_given = int(s.get("food_given", 0))
        if "conditions" in s:
            self.set_conditions(**s["conditions"])


def describe_rules() -> str:
    return "; ".join(f"{n}: {(c.__doc__ or '').strip().splitlines()[0]}" for n, c in RULES.items())


__all__ = ["TeachingToolkit", "RULES", "Rule", "describe_rules", "clean_text", "clean_note", "clean_prefix",
           "WHERE"]
