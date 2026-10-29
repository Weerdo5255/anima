"""Teachers decide *how to teach*; the reflexes and the creature do the rest.

* :class:`ScriptedTeacher` — a fixed, competent policy.  Needs nothing,
  runs instantly; the baseline and the fallback when no LLM is available.
* :class:`OllamaTeacher`  — a local language model served by Ollama.  It sees
  a compact summary of the creature every few seconds and answers with a JSON
  decision (constrained by a JSON schema).  If the model is slow, the creature
  simply keeps living; if it fails or answers nonsense, the scripted teacher
  covers that turn.

Every decision, from either teacher, goes through :func:`validate`, which
enforces the lesson's rules (e.g. no free food during the "ask" lesson), so
a small model can be creative about *when* and *what* to say without being
able to break the experiment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..brain.presets import CONSONANTS, VOWELS
from .curriculum import LESSONS
from .ollama import OllamaClient, OllamaError
from .toolkit import RULES, WHERE, clean_note, clean_prefix, clean_text

TOOLS = ("wait", "say", "praise", "scold", "give_food", "set_rule", "clear_rule", "switch_lesson", "note")
MAX_ACTIONS = 3
STARVING = 0.85  # hunger at which any lesson allows feeding it
REVIEW_BELOW = 0.5  # the scripted teacher reviews foraging if a spot-check exam falls below this


@dataclass
class Decision:
    thought: str
    actions: list[dict[str, Any]]
    source: str = "scripted"
    seconds: float = 0.0
    rejected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        for a in self.actions:
            tool = a["tool"]
            if tool == "say":
                parts.append(f"say '{a['text']}'")
            elif tool == "give_food":
                parts.append(f"give_food({a['where']})")
            elif tool in ("set_rule", "clear_rule"):
                arg = a["rule"] + (f", prefix={a['prefix']}" if a.get("prefix") else "")
                parts.append(f"{tool}({arg})")
            elif tool == "switch_lesson":
                parts.append(f"switch_lesson({a['lesson']})")
            elif tool == "note":
                parts.append("note")
            else:
                parts.append(tool)
        return ", ".join(parts) or "wait"


# ----------------------------------------------------------------- validation
def validate(raw: Any, obs: dict[str, Any], source: str) -> Decision:
    """Turn a raw (possibly sloppy) decision into a safe one for this lesson."""
    rejected: list[str] = []
    if not isinstance(raw, dict):
        return Decision("(unreadable decision)", [{"tool": "wait"}], source, rejected=["not an object"])
    thought = clean_note(raw.get("thought", ""), 240) or "(no thought given)"
    acts = raw.get("actions", [])
    if isinstance(acts, dict):
        acts = [acts]
    if not isinstance(acts, list):
        acts = []
    info = obs.get("lesson") or {}
    current = info.get("name", "")
    active = info.get("reviewing") or current  # the lesson whose rules apply right now
    allowed_rules = set(info.get("rules_you_may_use", ()))
    food_allowed = bool(info.get("you_may_give_food", False))
    starving = (obs.get("body", {}).get("hunger") or 0.0) >= STARVING
    graduated = set(info.get("graduated", []))
    out: list[dict[str, Any]] = []
    used = set()
    for a in acts[: MAX_ACTIONS * 2]:
        if not isinstance(a, dict):
            continue
        tool = str(a.get("tool", "")).strip().lower()
        if tool not in TOOLS:
            rejected.append(f"unknown tool {tool!r}")
            continue
        if tool in used and tool not in ("set_rule", "clear_rule"):
            continue  # one of each per turn
        if tool == "say":
            text = clean_text(a.get("text", ""), 30)
            if not text:
                continue
            out.append({"tool": "say", "text": text})
        elif tool in ("praise", "scold", "wait"):
            out.append({"tool": tool})
        elif tool == "give_food":
            where = str(a.get("where", "ahead")).lower()
            where = where if where in WHERE else "ahead"
            if not (food_allowed or starving):
                rejected.append(f"give_food not allowed in lesson {active}")
                continue
            out.append({"tool": "give_food", "where": where})
        elif tool in ("set_rule", "clear_rule"):
            rule = str(a.get("rule", "")).lower()
            if rule not in RULES:
                rejected.append(f"unknown rule {rule!r}")
                continue
            if tool == "set_rule" and rule not in allowed_rules:
                rejected.append(f"rule {rule} not allowed in lesson {active}")
                continue
            act = {"tool": tool, "rule": rule}
            if tool == "set_rule" and rule in ("reward_syllable", "feed_on_call"):
                act["prefix"] = clean_prefix(a.get("prefix") or obs.get("target_sound", "m"))
            out.append(act)
        elif tool == "switch_lesson":
            name = str(a.get("lesson", "")).lower()
            if not name or (name != current and name not in graduated):
                rejected.append(f"cannot switch to lesson {name!r} (not graduated yet)")
                continue
            out.append({"tool": "switch_lesson", "lesson": name})
        elif tool == "note":
            text = clean_note(a.get("text", ""), 160)
            if text:
                out.append({"tool": "note", "text": text})
        used.add(tool)
        if len(out) >= MAX_ACTIONS:
            break
    if not out:
        out = [{"tool": "wait"}]
    return Decision(thought, out, source, rejected=rejected)


# ----------------------------------------------------------------- teachers
class Teacher:
    name = "teacher"

    def bind(self, lessons: Any, n_neurons: int = 1700) -> None:
        """Called by the tutor with its curriculum's lessons and the brain's size."""

    def decide(self, obs: dict[str, Any]) -> Decision:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        return {}


class ScriptedTeacher(Teacher):
    """A simple, sensible teacher that follows each lesson's tips."""

    name = "scripted"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self._last_cons = ""
        self._last_review = -1e9

    def decide(self, obs: dict[str, Any]) -> Decision:
        if obs.get("state") != "awake":
            return Decision("It is asleep; let it sleep.", [{"tool": "wait"}], self.name)
        if (obs.get("body", {}).get("hunger") or 0.0) >= STARVING:
            return validate({"thought": "It is starving; feed it, whatever the lesson.",
                             "actions": [{"tool": "give_food", "where": "ahead"}]}, obs, self.name)
        info = obs["lesson"]
        lesson = info.get("reviewing") or info["name"]
        exam = (obs.get("exams") or {}).get("forage") or {}
        forage = exam.get("score")
        age = float(obs.get("age_minutes") or 0.0)
        if (not info.get("reviewing") and "forage" in info.get("graduated", []) and lesson != "forage"
                and forage is not None and exam.get("minutes_ago", 99) < 2.0 and forage < REVIEW_BELOW
                and age - self._last_review >= 10.0):
            self._last_review = age
            return validate({"thought": f"Its foraging exam slipped to {forage:.2f}; a short review.",
                             "actions": [{"tool": "switch_lesson", "lesson": "forage"}]}, obs, self.name)
        rules = " ".join(obs.get("active_rules", []))
        target = obs.get("target_sound", "m")
        acts: list[dict[str, Any]] = []
        for rule in info.get("default_rules", ()):
            if rule not in rules:  # make sure the lesson's reflexes are on
                acts.append({"tool": "set_rule", "rule": rule, "prefix": target})
        if lesson == "babble" and self.rng.random() < 0.25:
            acts.append({"tool": "say", "text": f"{target}a {target}a"})
            thought = "Model the target sound now and then; the reflex rewards it when it says it."
        elif lesson == "echo":
            options = [c for c in CONSONANTS if c not in (self._last_cons, "h", target)]
            c = str(self.rng.choice(options))
            v = str(self.rng.choice(VOWELS))
            self._last_cons = c
            acts.append({"tool": "say", "text": f"{c}{v} {c}{v}"})
            thought = f"Offer a new sound ({c}) to copy; the reflex praises a copy."
        elif lesson == "forage":
            thought = "Let it practise; the guide keeps food within reach."
        elif lesson == "ask":
            thought = "Only its call brings food now; wait for it to ask."
        else:
            thought = "Watch it look after itself."
        if not acts:
            acts = [{"tool": "wait"}]
        return validate({"thought": thought, "actions": acts}, obs, self.name)


SYSTEM_PROMPT = """You are the teacher of a newborn artificial creature. Its brain is a simulated network of about NEURONS_HERE spiking neurons that learns only from experience, through dopamine released by food and by your attention. It is NOT a language model and understands no words. It can only:
- move (forward, turn left, turn right) in a small arena where food has a smell,
- babble syllables made of one consonant and one vowel ("ba", "mi", "to"),
- hear sounds (your words reach it as a stream of phonemes),
- feel hunger, fatigue, boredom and loneliness, and sleep at night.
It lives on its own clock and never waits for you. You get a short report every few seconds. If the report has "its_thoughts", those are lines the framework reads out of its brain activity (feelings, what it notices, what it is inclined to do, rewards better or worse than it expected); they are not words it knows.

How it learns: a reward (food, or your smile) strengthens whatever it did in the last second or so. Talking to it or smiling also relieves its loneliness, which is rewarding in itself, so attention given at random teaches random things. Precise, fast rewards are delivered by reflex rules that you switch on; you decide the strategy. Graduation from a lesson is measured automatically, partly by short exams given to a copy of the creature (results under "exams"; a skill that slips can be reviewed with switch_lesson).

Tools (use 0 to 3 actions per turn; doing nothing is often best):
{"tool": "wait"}
{"tool": "say", "text": "ma ma"}  - speak; use one or two short syllables, never sentences
{"tool": "praise"}  - smile at it, only right after something good
{"tool": "scold"}  - frown at it, rarely
{"tool": "give_food", "where": "ahead" | "left" | "right" | "near"}  - only if the lesson allows it, or if it is starving
{"tool": "set_rule", "rule": RULE, "prefix": "m"}  - switch a reflex rule on (prefix = the target sound)
{"tool": "clear_rule", "rule": RULE}  - switch a reflex rule off
{"tool": "switch_lesson", "lesson": LESSON}  - go back to review a lesson it already passed
{"tool": "note", "text": "..."}  - a short note to your future self

Reflex rules (RULE) - only those the current lesson allows can be switched on:
RULES_HERE

Lessons (taken in order; you cannot skip ahead):
LESSONS_HERE

Examples of good turns:
{"thought": "It keeps missing food on its left; give it an easy one there.", "actions": [{"tool": "give_food", "where": "left"}]}
{"thought": "Model the target sound once, then let the reflex reward it.", "actions": [{"tool": "say", "text": "ma ma"}]}
{"thought": "Offer a new consonant to copy.", "actions": [{"tool": "say", "text": "bo bo"}]}
{"thought": "It is not hungry yet; the call only works when it is.", "actions": [{"tool": "wait"}]}

Answer with one JSON object only: {"thought": "one short sentence", "actions": [ ... ]}"""


def _system_prompt(lessons: Any = LESSONS, n_neurons: int = 1700) -> str:
    def doc(c: type) -> str:
        return (c.__doc__ or "").strip().splitlines()[0].replace("``", "")

    rules = "\n".join(f"- {n}: {doc(c)}" for n, c in RULES.items())
    listing = "\n".join(f"{i + 1}. {lesson.name}: {lesson.goal}" for i, lesson in enumerate(lessons))
    approx = f"{int(round(n_neurons, -2)):,}" if n_neurons >= 1000 else str(n_neurons)
    return (SYSTEM_PROMPT.replace("RULES_HERE", rules).replace("LESSONS_HERE", listing)
            .replace("NEURONS_HERE", approx))


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "actions": {
            "type": "array",
            "maxItems": MAX_ACTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": list(TOOLS)},
                    "text": {"type": "string"},
                    "where": {"type": "string", "enum": list(WHERE)},
                    "rule": {"type": "string", "enum": list(RULES)},
                    "prefix": {"type": "string"},
                    "lesson": {"type": "string"},
                },
                "required": ["tool"],
            },
        },
    },
    "required": ["thought", "actions"],
}


_SCRIPTED = object()  # default fallback marker


class OllamaTeacher(Teacher):
    """A local LLM (via Ollama) as the teacher, with a scripted fallback.

    ``fallback`` covers any turn the model fails (unreachable, error, no
    usable JSON); pass ``fallback=None`` to just wait those turns instead.
    After ``max_failures`` failures in a row the model is given a rest of
    ``retry_after_s`` seconds (the fallback teaches meanwhile), so a crashed
    or hung server does not stall every turn.
    """

    name = "ollama"

    def __init__(self, client: OllamaClient | None = None, fallback: Any = _SCRIPTED,
                 model: str | None = None, host: str | None = None, max_failures: int = 3,
                 retry_after_s: float = 60.0) -> None:
        self.client = client or OllamaClient(model=model or "llama3.2", host=host)
        self.fallback: Teacher | None = ScriptedTeacher() if fallback is _SCRIPTED else fallback
        self.system = _system_prompt()
        self.max_failures = max_failures
        self.retry_after_s = retry_after_s
        self._rest_until = 0.0
        self.calls = 0
        self.failures = 0
        self.skipped = 0
        self.consecutive_failures = 0
        self.total_seconds = 0.0
        self.last_error = ""

    def bind(self, lessons: Any, n_neurons: int = 1700) -> None:
        self.system = _system_prompt(lessons, n_neurons)
        if self.fallback is not None:
            self.fallback.bind(lessons, n_neurons)

    def messages(self, obs: dict[str, Any]) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": "Report:\n" + json.dumps(obs, separators=(",", ":"))
                 + "\nWhat do you do now? JSON only."}]

    def _cover(self, obs: dict[str, Any], why: str, t0: float) -> Decision:
        if self.fallback is None:
            return Decision(f"(LLM {why})", [{"tool": "wait"}], "error", seconds=time.perf_counter() - t0)
        d = self.fallback.decide(obs)
        d.source = "fallback"
        d.seconds = time.perf_counter() - t0
        return d

    def decide(self, obs: dict[str, Any]) -> Decision:
        t0 = time.perf_counter()
        if time.monotonic() < self._rest_until:
            self.skipped += 1
            return self._cover(obs, "resting after repeated failures", t0)
        self.calls += 1
        try:
            raw, res = self.client.chat_json(self.messages(obs), schema=DECISION_SCHEMA)
            d = validate(raw, obs, f"ollama:{self.client.model}")
            d.seconds = time.perf_counter() - t0
            self.total_seconds += d.seconds
            self.consecutive_failures = 0
            return d
        except (OllamaError, ValueError) as exc:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = str(exc)
            if self.consecutive_failures >= self.max_failures:
                self._rest_until = time.monotonic() + self.retry_after_s
                self.consecutive_failures = 0
            return self._cover(obs, f"failed: {exc}", t0)

    def stats(self) -> dict[str, Any]:
        ok = self.calls - self.failures
        return {"model": self.client.model, "host": self.client.host, "calls": self.calls,
                "failures": self.failures, "skipped_while_resting": self.skipped,
                "last_error": self.last_error,
                "mean_seconds": round(self.total_seconds / ok, 2) if ok else None}


__all__ = ["Decision", "Teacher", "ScriptedTeacher", "OllamaTeacher", "validate", "DECISION_SCHEMA",
           "SYSTEM_PROMPT", "TOOLS", "STARVING"]
