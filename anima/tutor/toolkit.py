"""What a teacher can do to the creature — deliberately, and by reflex.

Reinforcement only works if it is *contingent*: the smile has to come within
about a second of the behaviour it rewards (that is the window of the
creature's eligibility traces).  An LLM takes seconds to think, so teaching
is split in two, the way a human teacher's is:

* **deliberate actions** chosen by the teacher every few seconds: say a word,
  give pleasure or pain (for what it just said or did), smile, frown, put
  food somewhere, and switch reflex rules on or off;
* **reflex rules** that run on the creature's own clock (every 10 ms tick)
  and respond within half a second: reward the target syllable, feed it
  when it calls while hungry, keep food within reach while it is learning to
  forage, reward it when it repeats the teacher's word.

The reflexes reward with **pleasure** (``reinforcer="pleasure"``, the
default): a primary reinforcer that teaches more strongly than a smile and
carries credit for exactly the syllables it rewards (the creature recalls
them while the dopamine is there; see :mod:`anima.brain.hedonic`).  A smile
also relieves loneliness, so a frown from a lonely creature's teacher can
even be rewarding; pleasure and pain have no such side effects.
``reinforcer="smile"`` teaches as anima 0.5 did.

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
from .skills import HUNGRY, SATED, SkillMeter, WordTrial, first_consonant

WHERE = tuple(OFFER_PLACES)  # ahead, left, right, near
REINFORCERS = ("pleasure", "smile")
CREDIT_FOR = ("reply", "speech", "movement", "now")  # what a teacher's pleasure or pain can be for
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

    def on_syllable(self, t: float, text: str, act: int | None = None) -> None:
        """Its syllable ``text`` at ``t`` (``act``: the brain's id for it)."""

    def on_tick(self, t: float) -> None:
        pass

    def on_reply(self, t: float, trial: WordTrial) -> None:
        """Its reply to something the teacher said is complete (``trial.result``)."""

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class RewardSyllable(Rule):
    """Reward (pleasure, or a smile) shortly after every syllable starting with ``prefix``."""

    name = "reward_syllable"

    def __init__(self, kit, prefix: str = "m", delay: float = 0.5, refractory: float = 2.0,
                 strength: float = 0.5) -> None:
        super().__init__(kit, prefix=clean_prefix(prefix))
        self.delay, self.refractory, self.strength = delay, refractory, strength

    def on_syllable(self, t, text, act=None):
        if text.startswith(self.params["prefix"]) and t - self.last >= self.refractory:
            self.last = t
            self.fired += 1
            acts = None if act is None else [act]
            self.kit.schedule(t + self.delay, lambda: self.kit._reward(self.strength, acts),
                              f"{self.kit.reward_word(self.strength)} for '{text}' (reflex)")


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

    def on_syllable(self, t, text, act=None):
        a = self.kit.agent
        if not (text.startswith(self.params["prefix"]) and a.brain.awake):
            return
        hunger = a.brain.drives.hunger
        acts = None if act is None else [act]
        if hunger >= self.params["min_hunger"] and t - self.last >= self.refractory:
            self.last = t
            self.fired += 1
            self.kit.schedule(t + self.delay, lambda: self.kit._place_food("ahead"), "feed (answering its call)")
            if self.params["praise"]:
                self.kit.schedule(t + self.delay, lambda: self.kit._reward(0.4, acts),
                                  f"{self.kit.reward_word(0.4)} (reflex)")
        elif (self.params["frown_when_full"] and hunger < self.full_below
              and t - self.last_frown >= self.refractory):
            # "you've just eaten": the call is for when it is hungry
            self.last_frown = t
            self.frowned += 1
            self.kit.schedule(t + self.delay, lambda: self.kit._punish(0.3, acts, self.frown_text),
                              f"{self.kit.punish_word(0.3)} (it called while full)")


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

    def on_syllable(self, t, text, act=None):
        trial = self.kit._echo_trial
        if trial is None or trial.get("end") is None or t < trial["end"]:
            return
        self.kit._echo_trial = None
        if t - trial["end"] <= self.window and text.startswith(trial["target"]):
            self.fired += 1
            if self.kit._word is not None:
                self.kit._word.rewarded = f"{self.kit.reward_word(0.5)} by reflex"
            acts = None if act is None else [act]
            self.kit.schedule(t + 0.2, lambda: self.kit._reward(0.5, acts),
                              f"{self.kit.reward_word(0.5)} (copied me)")


class RewardImitation(Rule):
    """After the teacher says a word, reward it for repeating it: a little for the right first sound, more for the right syllable, most for the whole word."""

    name = "reward_imitation"

    def __init__(self, kit, consonant: float = 0.3, syllable: float = 0.5, word: float = 1.0,
                 wrong: float = 0.0) -> None:
        super().__init__(kit)
        self.grades = {"consonant": float(consonant), "syllable": float(syllable), "word": float(word)}
        self.wrong = float(wrong)  # pain for a reply that repeats nothing (0: none)
        self.counts = {"word": 0, "syllable": 0, "consonant": 0, "none": 0}

    def on_reply(self, t, trial):
        res = trial.result or {}
        if not res.get("answered") or not trial.target:
            return
        level = res["level"]
        self.counts[level if level in self.counts else "none"] += 1
        x = self.grades.get(level, 0.0)
        heard = " ".join(trial.target)
        said = " ".join(trial.reply)
        if x > 0:
            self.fired += 1
            how = {"word": "the whole word", "syllable": "the first syllable",
                   "consonant": "the first sound"}[level]
            self.kit._reward(x, trial.acts or None)
            trial.rewarded = f"{self.kit.reward_word(x)} by reflex"
            self.kit._log(t, f"{self.kit.reward_word(x)}: it repeated {how} ('{heard}' -> '{said}')")
        elif self.wrong > 0:
            self.kit._punish(self.wrong, trial.acts or None)
            trial.rewarded = f"{self.kit.punish_word(self.wrong)} by reflex"
            self.kit._log(t, f"{self.kit.punish_word(self.wrong)}: '{heard}' -> '{said}'")


RULES: dict[str, type[Rule]] = {r.name: r for r in
                                (RewardSyllable, FeedOnCall, GuideFood, EchoBack, RewardEcho, RewardImitation)}


# ======================================================================= toolkit
class TeachingToolkit:
    def __init__(self, agent: Any, meter: SkillMeter, rng_seed: int = 0, reinforcer: str = "pleasure") -> None:
        if reinforcer not in REINFORCERS:
            raise ValueError(f"reinforcer must be one of {REINFORCERS}, not {reinforcer!r}")
        self.agent = agent
        self.meter = meter
        self.reinforcer = reinforcer
        self.rng = np.random.default_rng(rng_seed)
        self._rules: dict[str, Rule] = {}
        self._lock = threading.Lock()
        self._pending: list[tuple[float, int, Callable[[], None], str]] = []
        self._seq = 0
        self._echo_trial: dict[str, Any] | None = None
        self._word: WordTrial | None = None
        self.feelings = {"pleasure": 0, "pain": 0}  # given by the teacher or its reflexes
        self.trials: deque[dict[str, Any]] = deque(maxlen=20)  # recent "I say a word, you answer" exchanges
        self.trial_count = 0
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

    def pleasure(self, amount: float = 0.6, credit: str = "reply") -> None:
        """Give it pleasure for its answer to the teacher's last word
        (``credit="reply"``, even if it has said other things since), for
        what it said last (``"speech"``), did (``"movement"``) or nothing in
        particular (``"now"``)."""
        self.agent.call_soon(lambda ag: self._feel("pleasure", amount, self._credit(credit), "teacher"))

    def pain(self, amount: float = 0.4, credit: str = "reply") -> None:
        """Give it pain for its reply, its last words or its last movement (see :meth:`pleasure`)."""
        self.agent.call_soon(lambda ag: self._feel("pain", amount, self._credit(credit), "teacher"))

    def _credit(self, credit: str) -> Any:
        if credit != "reply":
            return credit
        last = self.trials[-1] if self.trials else None
        return list(last["acts"]) if last and last.get("acts") else "now"

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

    def _feel(self, kind: str, amount: float, credit: Any, source: str = "reflex") -> dict[str, Any]:
        """Pain or pleasure now (agent thread)."""
        self.feelings[kind] = self.feelings.get(kind, 0) + 1
        return self.agent.brain.feel(kind, float(amount), f"tutor:{source}", credit)

    def reward_word(self, strength: float) -> str:
        return f"pleasure {strength:.1f}" if self.reinforcer == "pleasure" else "praise"

    def punish_word(self, strength: float) -> str:
        return f"pain {strength:.1f}" if self.reinforcer == "pleasure" else "frown"

    def _reward(self, strength: float, acts: list[int] | None = None) -> None:
        """The reflexes' reward: pleasure for ``acts`` (its syllables), or a smile."""
        if self.reinforcer == "pleasure":
            self._feel("pleasure", strength, acts if acts else "now")
        else:
            self._hear(":)")

    def _punish(self, strength: float, acts: list[int] | None = None, frown: str = ":(") -> None:
        if self.reinforcer == "pleasure":
            self._feel("pain", strength, acts if acts else "now")
        else:
            self._hear(frown)

    def _say(self, text: str) -> None:
        self.agent.hear(text)
        self.meter.note_teacher_said(text)
        c = first_consonant(text)
        self._echo_trial = None if c is None else {"target": c, "end": None}
        w = WordTrial(text, self.agent.t_ms / 1000.0)
        self._word = w if w.target else None
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
            r.on_syllable(e["t"], e["text"], e.get("act"))
        w = self._word
        if w is not None and w.syllable(e["t"], e["text"], e.get("act")):
            self._reply_done(e["t"], w, rules)

    def _reply_done(self, t: float, w: WordTrial, rules: list[Rule]) -> None:
        self._word = None
        for r in rules:
            r.on_reply(t, w)
        res = w.result or {}
        self.trial_count += 1
        self.trials.append({"n": self.trial_count, "t": round(t, 2), "you_said": " ".join(w.target),
                            "it_replied": " ".join(w.reply), "acts": list(w.acts),
                            "match": res.get("level", "none") if res.get("answered") else "no answer",
                            "rewarded": w.rewarded})

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
        w = self._word
        if w is not None:
            if self._was_hearing and not hearing:
                w.heard_end(t)
            if w.tick(t):
                with self._lock:
                    rules_now = list(self._rules.values())
                self._reply_done(t, w, rules_now)
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
                "food_given": self.food_given, "conditions": self.conditions(),
                "reinforcer": self.reinforcer, "feelings": dict(self.feelings)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.clear_rules()
        for r in s.get("rules", []):
            if r["name"] in RULES:
                self.set_rule(r["name"], **r.get("params", {}))
        self.journal.extend((float(t), str(x)) for t, x in s.get("journal", []))
        self.food_given = int(s.get("food_given", 0))
        self.feelings.update({k: int(v) for k, v in s.get("feelings", {}).items()})
        if "conditions" in s:
            self.set_conditions(**s["conditions"])


def describe_rules() -> str:
    return "; ".join(f"{n}: {(c.__doc__ or '').strip().splitlines()[0]}" for n, c in RULES.items())


__all__ = ["TeachingToolkit", "RULES", "Rule", "describe_rules", "clean_text", "clean_note", "clean_prefix",
           "WHERE", "REINFORCERS", "CREDIT_FOR"]
