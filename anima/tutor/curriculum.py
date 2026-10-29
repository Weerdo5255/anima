"""The curriculum: four lessons from helpless to independent.

Each lesson names the reflex rules that run by default, which rules the
teacher may switch on, a minimum amount of *awake* practice, and an
objective graduation test — on the passive skill measurements, or on an exam
taken by a clone of the creature (see :mod:`.exams`).  Graduation is decided
by these tests, not by the teacher's opinion, so a weak or overly optimistic
language model cannot skip ahead.  A lesson that is still not passed after
``max_minutes`` of awake practice is left "not mastered" and the curriculum
moves on (it can be reviewed later), so training never gets stuck.

1. forage       find food alone (scaffolding keeps food within reach, and
                it is kept keen — mildly hungry — as animal trainers do)
2. echo         copy the consonant it just heard (praise for copies; still
                kept a little hungry, so foraging stays in practice)
3. babble       make the target sound often (contingent smiles)
4. independence no teacher, no reflexes, no help: keep itself fed for eight
                awake minutes

Copying comes before the target sound on purpose: once one sound dominates
its babbling, the echoic trace of a sound it hears keeps getting paired with
the dominant sound it makes next, which blurs the hearing-to-speaking map
copying relies on (the echo rate fell from about 0.2 to 0.05-0.1 when the
order was the other way round).

:data:`ASK` — use the target sound to ask for food when hungry — is an
optional, experimental fifth lesson (``LESSONS_WITH_ASK``; ``anima train
--with-ask``).  With the default brain the creature learns to call for food,
but not to call *only* when hungry: dopamine is global and its frequent
"m" gets credited for unrelated rewards through the drives that are active
in every state (boredom, loneliness), so in our runs the hunger-specific
part stayed small.  See the README's honest limits.

Each lesson also sets the conditions the tutor maintains while it runs
(``energy_cap``: food restriction; ``wild_food``: whether natural food grows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .skills import SkillReport


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _ok(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


Exams = dict[str, dict[str, Any]]


def check_forage(r: SkillReport, ex: Exams) -> tuple[bool, str]:
    e = ex.get("forage")
    score = e["score"] if e else None
    ok = _ok(score) and score >= 0.60 and r.food_per_min >= 1.0
    return ok, (f"exam {_fmt(score) if e else 'not taken yet'} (need 0.60; chance 0.33), "
                f"food {r.food_per_min:.1f}/min (need 1.0)")


def check_babble(r: SkillReport, ex: Exams) -> tuple[bool, str]:
    ok = r.target_n >= 80 and _ok(r.target_share) and r.target_share >= 0.30
    return ok, f"target share {_fmt(r.target_share)} (need 0.30, n={r.target_n}/80)"


def check_echo(r: SkillReport, ex: Exams) -> tuple[bool, str]:
    need = max(0.15, 1.8 * r.echo_chance) if _ok(r.echo_chance) else 0.15
    ok = r.echo_n >= 30 and _ok(r.echo_rate) and r.echo_rate >= need
    return ok, f"echo rate {_fmt(r.echo_rate)} (need {need:.2f}, chance {_fmt(r.echo_chance)}, n={r.echo_n}/30)"


def check_ask(r: SkillReport, ex: Exams) -> tuple[bool, str]:
    e = ex.get("ask")
    if not e:
        return False, ("exam not taken yet (need: the target sound in >= 25% of its syllables when "
                       "hungry, and at least 15 points more often than when sated)")
    h, s_ = e["hungry"], e["sated"]
    ok = (e["n_hungry"] >= 15 and e["n_sated"] >= 15 and _ok(h) and _ok(s_)
          and h >= 0.25 and h - s_ >= 0.15)
    return ok, (f"exam: says it in {_fmt(h)} of syllables when hungry (need 0.25) vs {_fmt(s_)} when "
                f"sated (need 0.15 less; n={e['n_hungry']}/{e['n_sated']})")


def check_independence(r: SkillReport, ex: Exams) -> tuple[bool, str]:
    ok = (r.energy_minutes >= 8 and _ok(r.energy_mean) and r.energy_mean >= 0.35
          and _ok(r.energy_min) and r.energy_min >= 0.08)
    return ok, (f"energy mean {_fmt(r.energy_mean)} (need 0.35), min {_fmt(r.energy_min)} (need 0.08) "
                f"over {r.energy_minutes:.1f}/8 min")


@dataclass(frozen=True)
class Lesson:
    name: str
    title: str
    goal: str
    default_rules: tuple[str, ...]
    allowed_rules: tuple[str, ...]
    min_minutes: float
    check: Callable[[SkillReport, Exams], tuple[bool, str]]
    tips: str
    food_allowed: bool = False
    energy_cap: float | None = None  # keep it keen: energy held at or below this
    wild_food: bool = True  # natural food grows in the arena
    exam: str | None = None  # exam (in .exams) the check needs, taken once practice is done
    exam_every_min: float = 2.0  # awake minutes between exams
    measures: tuple[str, ...] = ()  # SkillMeter windows to start afresh with this lesson
    max_minutes: float | None = 45.0  # awake practice after which it moves on, not mastered

    def conditions(self) -> dict[str, Any]:
        return {"energy_cap": self.energy_cap, "wild_food": self.wild_food}


LESSONS: tuple[Lesson, ...] = (
    Lesson("forage", "Find food",
           "turn toward the smell of food and walk to it, without help",
           ("guide_food",), ("guide_food",), 5.0, check_forage,
           "guide_food keeps food within reach while it is hungry, and it is kept mildly hungry so "
           "food stays rewarding. You may also put food to its left or right to give it practice "
           "turning. Do not praise every step.", food_allowed=True, energy_cap=0.5, exam="forage"),
    Lesson("echo", "Copy me",
           "start its reply with the consonant it just heard",
           ("reward_echo",), ("reward_echo", "echo_back"), 5.0, check_echo,
           "Say one short repeated syllable each turn ('da da', 'bo bo', 'ki ki') with a different "
           "consonant each time, and leave out the target sound (that comes in the next lesson); "
           "reward_echo smiles when it copies. Let it answer before speaking again.",
           energy_cap=0.6, measures=("echo",)),
    Lesson("babble", "Make the sound",
           "say syllables starting with the target sound often",
           ("reward_syllable",), ("reward_syllable", "echo_back"), 4.0, check_babble,
           "reward_syllable smiles right after each target syllable. You can model the sound by "
           "saying it (e.g. 'ma ma') now and then. Never smile at other sounds.", energy_cap=0.6,
           measures=("target",)),
    Lesson("independence", "Live on its own",
           "keep itself fed with no help at all",
           (), (), 10.0, check_independence,
           "Step back: no reflexes, no food from you (unless it is starving). Watch, and talk to it "
           "now and then if you like.", measures=("energy",)),
)

LESSON_NAMES = tuple(lesson.name for lesson in LESSONS)

#: optional, experimental: ask for food with the target sound (see module docs)
ASK = Lesson("ask", "Ask for food",
           "say the target sound when hungry, because that brings food",
           ("feed_on_call",), ("feed_on_call",), 5.0, check_ask,
           "No food grows in the arena now: feed_on_call answers its call with food, only when it "
           "is hungry, and frowns at it if it calls when full. Do NOT give food any other way, or "
           "the call means nothing.", wild_food=False, exam="ask", exam_every_min=3.0,
           measures=("ask",), max_minutes=30.0)
LESSONS_WITH_ASK: tuple[Lesson, ...] = LESSONS[:3] + (ASK,) + LESSONS[3:]
#: every built-in lesson by name (used to restore a saved curriculum)
ALL_LESSONS: dict[str, Lesson] = {lesson.name: lesson for lesson in LESSONS_WITH_ASK}


class Curriculum:
    def __init__(self, lessons: tuple[Lesson, ...] = LESSONS) -> None:
        self.lessons = lessons
        self.index = 0
        self.awake_s = 0.0  # awake practice in the current lesson
        self.graduated: list[dict[str, Any]] = []
        self.finished = False
        self.started_at = 0.0

    @property
    def current(self) -> Lesson:
        return self.lessons[min(self.index, len(self.lessons) - 1)]

    def lesson(self, name: str) -> Lesson:
        return next(lesson for lesson in self.lessons if lesson.name == name)

    def graduated_names(self) -> list[str]:
        """Lessons it has been through (passed or not) — these can be reviewed."""
        return [g["lesson"] for g in self.graduated]

    def mastered_names(self) -> list[str]:
        return [g["lesson"] for g in self.graduated if g.get("mastered", True)]

    def practice(self, dt_s: float, awake: bool) -> None:
        if awake:
            self.awake_s += dt_s

    def evaluate(self, report: SkillReport, exams: Exams | None = None,
                 previous: Exams | None = None) -> tuple[bool, str]:
        """Is the current lesson passed?  An exam-based test must pass on the
        last two exams (``exams`` and ``previous``), not on one lucky one."""
        lesson = self.current
        ok, progress = lesson.check(report, exams or {})
        if lesson.exam and ok:
            ok = previous is not None and lesson.exam in previous and lesson.check(report, previous)[0]
            if not ok:
                progress += "; must pass the next exam too"
        minutes = self.awake_s / 60.0
        enough = minutes >= lesson.min_minutes
        if not enough:
            progress += f"; practised {minutes:.1f}/{lesson.min_minutes:.0f} awake min"
        return ok and enough and not self.finished, progress

    def overdue(self) -> bool:
        """Has the current lesson had its maximum practice without being passed?"""
        m = self.current.max_minutes
        return m is not None and self.awake_s >= m * 60.0 and not self.finished

    def graduate(self, t_s: float, report: SkillReport, exams: Exams | None = None,
                 mastered: bool = True) -> Lesson | None:
        """Record the end of the current lesson (passed, or left unmastered after its
        maximum practice); return the next lesson (None when done)."""
        self.graduated.append({"lesson": self.current.name, "t": round(t_s, 1),
                               "awake_minutes": round(self.awake_s / 60.0, 2), "mastered": mastered,
                               "skills": report.compact(), "exams": dict(exams or {})})
        if self.index + 1 >= len(self.lessons):
            self.finished = True
            return None
        self.index += 1
        self.awake_s = 0.0
        self.started_at = t_s
        return self.current

    def status(self, report: SkillReport, exams: Exams | None = None,
               previous: Exams | None = None) -> dict[str, Any]:
        ready, progress = self.evaluate(report, exams, previous)
        lesson = self.current
        return {"name": lesson.name, "title": lesson.title, "goal": lesson.goal,
                "number": f"{self.index + 1} of {len(self.lessons)}", "progress": progress,
                "ready_to_graduate": ready, "awake_minutes": round(self.awake_s / 60.0, 1),
                "graduated": self.graduated_names(), "finished": self.finished}

    def state_dict(self) -> dict[str, Any]:
        return {"lessons": [lesson.name for lesson in self.lessons], "index": self.index,
                "awake_s": self.awake_s, "graduated": self.graduated,
                "finished": self.finished, "started_at": self.started_at}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.index = int(s.get("index", 0))
        self.awake_s = float(s.get("awake_s", 0.0))
        self.graduated = list(s.get("graduated", []))
        self.finished = bool(s.get("finished", False))
        self.started_at = float(s.get("started_at", 0.0))
