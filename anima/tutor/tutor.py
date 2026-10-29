"""The tutor: a teacher (LLM or scripted) raising an anima creature.

    creature thread  ── lives in real time, never waits ──────────────────────►
                         ▲ reflex rules act within ~0.5 s (tick hooks)
    tutor thread     ── every few seconds: observe → teacher.decide (may take
                         seconds for an LLM) → validate → act ────────────────►

The tutor owns a :class:`SkillMeter` (passive measurement), a
:class:`TeachingToolkit` (actions and reflexes) and a :class:`Curriculum`.
When the current lesson's graduation test passes, it moves to the next one
and switches the reflexes accordingly.  When the last lesson is passed, the
creature is on its own — a trained brain you can run with ``anima live``.

The teacher may send the creature back to review a lesson it has already
passed (for a few minutes); progress in the current lesson pauses meanwhile.

Its state (curriculum progress, skill history, journal) is saved next to the
brain file as ``<brain>.tutor.json`` so training can continue over many
sessions, building skills slowly.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .curriculum import Curriculum
from .exams import EXAMS, summarize
from .skills import SkillMeter
from .teachers import Decision, ScriptedTeacher, Teacher
from .toolkit import TeachingToolkit


class Tutor:
    def __init__(self, agent: Any, teacher: Teacher | None = None, target: str = "m",
                 think_every_s: float = 8.0, printer: Callable[[str], None] | None = print,
                 verbose: bool = True, review_minutes: float = 3.0, spot_check_min: float = 12.0,
                 hold_s: float = 60.0, curriculum: Curriculum | None = None) -> None:
        from ..brain.presets import CONSONANTS
        if target not in CONSONANTS:
            raise ValueError(f"target must be one of the consonants {CONSONANTS}, not {target!r}")
        self.agent = agent
        self.teacher = teacher or ScriptedTeacher()
        self.target = target
        self.think_every_s = think_every_s
        self.review_minutes = review_minutes
        self.reviewing: str | None = None  # a passed lesson being revisited
        self.review_until = 0.0
        self.spot_check_min = spot_check_min  # re-test passed exam skills this often
        self.hold_s = hold_s  # a lesson's test must keep passing this long to graduate
        self._ready_since: float | None = None
        self.exams: dict[str, dict[str, Any]] = {}  # latest result of each exam
        self.exam_log: deque[dict[str, Any]] = deque(maxlen=100)  # all recent results, oldest first
        self.printer = printer
        self.verbose = verbose
        self.meter = SkillMeter(agent, target)
        self.toolkit = TeachingToolkit(agent, self.meter)
        self.curriculum = curriculum or Curriculum()
        self.teacher.bind(self.curriculum.lessons, n_neurons=int(agent.brain.net.N))
        self.history: deque[dict[str, Any]] = deque(maxlen=200)
        self.milestones: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_counts = agent.body.motor.counts.copy() if agent.body.motor else None
        self._last_meals = 0
        self._last_obs_t = agent.t_ms / 1000.0
        self._asleep_logged = False
        agent.add_tick_hook(self._on_tick)
        self._apply_lesson(self.curriculum.current.name, announce=False)

    # ------------------------------------------------------------ helpers
    def _say(self, msg: str) -> None:
        if self.printer is not None:
            self.printer(msg)

    def _on_tick(self, agent: Any) -> None:
        self.curriculum.practice(agent.tick_ms / 1000.0, agent.brain.awake and self.reviewing is None)

    def _apply_lesson(self, name: str, announce: bool = True) -> None:
        """Switch the reflexes and training conditions to those of lesson ``name``."""
        lesson = self.curriculum.lesson(name)
        self.toolkit.clear_rules()
        for rule in lesson.default_rules:
            params = {"prefix": self.target} if rule in ("reward_syllable", "feed_on_call") else {}
            self.toolkit.set_rule(rule, **params)
        self.toolkit.set_conditions(lesson.energy_cap, lesson.wild_food)
        if announce:
            number = [x.name for x in self.curriculum.lessons].index(name) + 1
            self._say(f"   ── lesson {number}: {lesson.title} — {lesson.goal}")

    def _start_review(self, name: str) -> None:
        self.reviewing = name
        self.review_until = self.agent.t_ms / 1000.0 + self.review_minutes * 60.0
        self._apply_lesson(name, announce=False)
        self.toolkit.note(f"reviewing lesson {name}")
        self._say(f"   ── reviewing lesson '{name}' for {self.review_minutes:.0f} min")

    def _end_review(self) -> None:
        if self.reviewing is None:
            return
        self.reviewing = None
        self._apply_lesson(self.curriculum.current.name, announce=False)
        self._say(f"   ── back to lesson '{self.curriculum.current.name}'")

    def clock(self) -> str:
        return self.agent.latest.get("clock", "--:--")

    # ------------------------------------------------------------ exams
    def take_exam(self, name: str) -> dict[str, Any]:
        """Examine a clone of the creature (the creature itself carries on)."""
        res = EXAMS[name](self.agent, target=self.target)
        res.update({"t": round(self.agent.t_ms / 1000.0, 1), "clock": self.clock(),
                    "lesson_index": self.curriculum.index,
                    "lesson_awake_s": round(self.curriculum.awake_s, 1)})
        self.exams[name] = res
        self.exam_log.append({"exam": name, **res})
        self._say(f"   ✎ [{res['clock']}] exam '{name}': {summarize(name, res)}")
        return res

    def lesson_exams(self, back: int = 0) -> dict[str, dict[str, Any]]:
        """Exam results taken during the current lesson: the latest of each
        (``back=0``) or the one before it (``back=1``)."""
        out: dict[str, dict[str, Any]] = {}
        seen: dict[str, int] = {}
        for e in reversed(self.exam_log):
            if e.get("lesson_index") != self.curriculum.index:
                continue
            k = e["exam"]
            if seen.get(k, 0) == back and k not in out:
                out[k] = e
            seen[k] = seen.get(k, 0) + 1
        return out

    def _exam_due(self) -> str | None:
        cur, lesson = self.curriculum, self.curriculum.current
        if self.reviewing is not None or cur.finished or not self.agent.brain.awake:
            return None
        if lesson.exam and cur.awake_s >= lesson.min_minutes * 60.0:
            last = self.lesson_exams().get(lesson.exam)
            if last is None or cur.awake_s - last["lesson_awake_s"] >= lesson.exam_every_min * 60.0:
                return lesson.exam
        now = self.agent.t_ms / 1000.0
        for g in cur.graduated_names():  # spot checks of skills passed earlier
            exam = cur.lesson(g).exam
            if exam and exam != lesson.exam:
                last = self.exams.get(exam)
                if last is None or now - last["t"] >= self.spot_check_min * 60.0:
                    return exam
        return None

    def _exams_view(self) -> dict[str, Any]:
        now = self.agent.t_ms / 1000.0
        out = {}
        for k, v in self.exams.items():
            keep = {x: v[x] for x in ("score", "hungry", "sated") if x in v}
            out[k] = {**{x: (None if isinstance(y, float) and np.isnan(y) else y) for x, y in keep.items()},
                      "minutes_ago": round((now - v["t"]) / 60.0, 1)}
        return out

    # ------------------------------------------------------------ observation
    def observe(self) -> dict[str, Any]:
        a = self.agent
        snap = a.latest
        now = a.t_ms / 1000.0
        report = self.meter.report()
        counts = a.body.motor.counts.copy() if a.body.motor else None
        did: dict[str, int] = {}
        if counts is not None and self._last_counts is not None:
            for label, n in zip(a.body.motor.labels, counts - self._last_counts):
                did[label] = int(n)
        self._last_counts = counts
        meals = a.body.stats.get("ate", 0)
        did["ate"] = int(meals - self._last_meals)
        self._last_meals = meals
        window = max(1.0, now - self._last_obs_t)
        self._last_obs_t = now
        heard = [d["text"] for t, k, d in list(a.log) if k == "heard" and now - t <= window]
        r = report.compact()
        lesson = self.curriculum.lesson(self.reviewing) if self.reviewing else self.curriculum.current
        status = self.curriculum.status(report, self.lesson_exams(), self.lesson_exams(back=1))
        d = snap.get("drives", {})
        m = snap.get("modulators", {})
        notes = [x for _, x in list(self.toolkit.journal) if x.startswith("note:")][-3:]
        recent = [x for _, x in list(self.toolkit.journal)[-6:] if not x.startswith("note:")]
        return {
            "clock": snap.get("clock"), "state": snap.get("stage", "awake"),
            "age_minutes": round(now / 60.0, 1), "seconds_since_last_report": round(window, 1),
            "body": {k: round(float(v), 2) for k, v in d.items()},
            "energy": round(float(a.body.energy), 2),
            "mood": {"dopamine": round(float(m.get("dopamine_tonic", 0)), 2),
                     "serotonin": round(float(m.get("serotonin", 0)), 2)},
            "it_said": self.meter.recent_syllables(window)[-10:],
            "it_did": did, "it_heard": heard[-4:],
            "skills": {"food_per_min": round(r["food_per_min"], 2),
                       "target_share": r["target_share"], "echo_rate": r["echo_rate"],
                       "echo_chance": r["echo_chance"], "asks_when_hungry": r["ask_hungry"],
                       "asks_when_full": r["ask_full"], "energy_mean": r["energy_mean"]},
            "exams": self._exams_view(),
            "lesson": {**status, "tips": lesson.tips, "reviewing": self.reviewing,
                       "rules_you_may_use": list(lesson.allowed_rules),
                       "default_rules": list(lesson.default_rules),
                       "you_may_give_food": lesson.food_allowed},
            "conditions": self.toolkit.conditions(),
            "target_sound": self.target,
            "active_rules": self.toolkit.active_rules(),
            "your_recent_actions": recent, "your_notes": notes,
        }

    # ------------------------------------------------------------ acting
    def execute(self, d: Decision) -> list[str]:
        done = []
        kit = self.toolkit
        for act in d.actions:
            tool = act["tool"]
            if tool == "say":
                kit.say(act["text"])
            elif tool == "praise":
                kit.praise()
            elif tool == "scold":
                kit.scold()
            elif tool == "give_food":
                if not kit.give_food(act["where"]):
                    continue
            elif tool == "set_rule":
                params = {"prefix": act["prefix"]} if "prefix" in act else {}
                kit.set_rule(act["rule"], **params)
            elif tool == "clear_rule":
                kit.clear_rule(act["rule"])
            elif tool == "switch_lesson":
                if act["lesson"] == self.curriculum.current.name:
                    self._end_review()
                elif act["lesson"] != self.reviewing:
                    self._start_review(act["lesson"])
            elif tool == "note":
                kit.note(act["text"])
            if tool != "wait":
                kit._log(self.agent.t_ms / 1000.0, d.summary() if tool != "note" else "note")
            done.append(tool)
        return done

    def _check_graduation(self) -> None:
        if self.reviewing is not None:
            return
        due = self._exam_due()
        if due is not None:
            self.take_exam(due)
        report = self.meter.report()
        exams = self.lesson_exams()
        ready, progress = self.curriculum.evaluate(report, exams, self.lesson_exams(back=1))
        now = self.agent.t_ms / 1000.0
        if not ready:
            self._ready_since = None
            if self.curriculum.overdue():  # never get stuck: move on, not mastered
                self._finish_lesson(report, exams, progress, mastered=False)
            return
        if self._ready_since is None:  # it must hold for a while, not just at one lucky moment
            self._ready_since = now
        if now - self._ready_since < self.hold_s:
            return
        self._finish_lesson(report, exams, progress, mastered=True)

    def _finish_lesson(self, report: Any, exams: dict[str, Any], progress: str, mastered: bool) -> None:
        self._ready_since = None
        old = self.curriculum.current
        minutes = self.curriculum.awake_s / 60.0
        nxt = self.curriculum.graduate(self.agent.t_ms / 1000.0, report, exams, mastered=mastered)
        ms = {"t": round(self.agent.t_ms / 1000.0, 1), "clock": self.clock(), "graduated": old.name,
              "mastered": mastered, "progress": progress}
        self.milestones.append(ms)
        if mastered:
            self._say(f"★ [{self.clock()}] graduated '{old.title}': {progress}")
        else:
            self._say(f"→ [{self.clock()}] moving on from '{old.title}' after {minutes:.0f} awake min, "
                      f"not mastered ({progress}); it can be reviewed later")
        if nxt is None:
            self.toolkit.clear_rules()
            self.toolkit.set_conditions(None, True)
            self._say("★ curriculum complete — it lives on its own now; the teacher steps away.")
        else:
            self.meter.clear(*nxt.measures)
            self._apply_lesson(nxt.name)

    def think_once(self) -> Decision:
        """One observe → decide → act cycle (blocking only the tutor's own thread)."""
        if self.reviewing is not None and self.agent.t_ms / 1000.0 >= self.review_until:
            self._end_review()
        self._check_graduation()
        if self.curriculum.finished:
            return Decision("Curriculum complete; observing.", [{"tool": "wait"}], "tutor")
        if not self.agent.brain.awake:
            if self.verbose and not self._asleep_logged:
                self._say(f"   [{self.clock()}] it is asleep — the teacher waits (replay consolidates the day)")
            self._asleep_logged = True
            return Decision("asleep", [{"tool": "wait"}], "tutor")
        self._asleep_logged = False
        obs = self.observe()
        d = self.teacher.decide(obs)
        self.execute(d)
        entry = {"t": round(self.agent.t_ms / 1000.0, 1), "clock": obs["clock"],
                 "lesson": obs["lesson"]["name"], "thought": d.thought, "actions": d.summary(),
                 "source": d.source, "seconds": round(d.seconds, 2), "rejected": d.rejected}
        self.history.append(entry)
        if self.verbose:
            lat = f" ({d.seconds:.1f}s)" if d.seconds >= 0.05 else ""
            rej = f"  [rejected: {'; '.join(d.rejected)}]" if d.rejected else ""
            self._say(f"   [{obs['clock']} {obs['lesson']['name']}] {d.source}{lat}: {d.thought} → "
                      f"{d.summary()}{rej}")
        return d

    # ------------------------------------------------------------ running
    def run(self, creature_seconds: float | None = None) -> None:
        """Teach (in this thread) while the creature lives in its own thread."""
        self._stop.clear()
        end = None if creature_seconds is None else self.agent.t_ms / 1000.0 + creature_seconds
        next_t = self.agent.t_ms / 1000.0
        while not self._stop.is_set():
            now = self.agent.t_ms / 1000.0
            if end is not None and now >= end:
                break
            if now < next_t:
                self._stop.wait(0.05)
                continue
            next_t = now + self.think_every_s
            try:
                self.think_once()
            except Exception as exc:  # teaching must never kill the creature
                self._say(f"   tutor error: {exc!r}")
            # after a slow turn (an exam, a slow model) leave it a moment
            # before deciding again, rather than acting twice in a row
            next_t = max(next_t, self.agent.t_ms / 1000.0 + 0.5 * self.think_every_s)

    def start(self) -> "Tutor":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(target=self.run, name="anima-tutor", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def train(self, minutes: float, speed: float = 1.0, realtime: bool = True,
              report_every_min: float = 5.0, stop_when_done: bool = True,
              checkpoint: Callable[[], None] | None = None) -> str:
        """Live and learn for up to ``minutes`` of creature time; returns the report.

        realtime=True paces the creature at ``speed`` x real time (the LLM sees
        the world change at a natural pace); False runs it as fast as the CPU
        allows (fine with the scripted teacher).  Stops early once the whole
        curriculum is passed (unless ``stop_when_done`` is False).
        ``checkpoint()`` (e.g. saving) is called with every progress report.
        """
        a = self.agent
        started_here = not a.alive
        if started_here:
            a.start(speed=speed, realtime=realtime)
        self.start()
        end = a.t_ms / 1000.0 + minutes * 60.0
        next_report = a.t_ms / 1000.0 + report_every_min * 60.0
        try:
            while a.t_ms / 1000.0 < end and not self._stop.is_set():
                time.sleep(0.2)
                if a.t_ms / 1000.0 >= next_report:
                    next_report += report_every_min * 60.0
                    self._say(self.progress_line())
                    if checkpoint is not None:
                        checkpoint()
                if self.curriculum.finished and stop_when_done:
                    break
        finally:
            self.stop()
            if started_here:
                a.stop()
        return self.report()

    # ------------------------------------------------------------ reporting
    def progress_line(self) -> str:
        report = self.meter.report()
        st = self.curriculum.status(report, self.lesson_exams(), self.lesson_exams(back=1))
        review = f" (reviewing '{self.reviewing}')" if self.reviewing else ""
        energy = "" if np.isnan(report.energy_mean) else f" | energy {report.energy_mean:.2f}"
        return (f"   ·· [{self.clock()}] lesson {st['number']} '{st['name']}'{review}: "
                f"{st['progress']}{energy}")

    def report(self) -> str:
        rep = self.meter.report()
        r = rep.compact()
        st = self.curriculum.status(rep, self.lesson_exams(), self.lesson_exams(back=1))
        lines = [f"Tutor report — creature age {self.agent.t_ms / 60000.0:.1f} min, teacher {self.teacher.name}"]
        for g in self.curriculum.graduated:
            if g.get("mastered", True):
                lines.append(f"  ✓ {g['lesson']:<13} passed at {g['t'] / 60:.1f} min "
                             f"after {g['awake_minutes']:.1f} awake min of practice")
            else:
                lines.append(f"  ✗ {g['lesson']:<13} not mastered; moved on at {g['t'] / 60:.1f} min "
                             f"after {g['awake_minutes']:.1f} awake min of practice")
        if not st["finished"]:
            lines.append(f"  … {st['name']:<13} in progress: {st['progress']}")

        def f(x: Any) -> str:
            return "n/a" if x is None else f"{x:.2f}"

        lines.append(f"  skills: food {r['food_per_min']:.1f}/min, '{self.target}' share {f(r['target_share'])}, "
                     f"echo {f(r['echo_rate'])} (chance {f(r['echo_chance'])}), energy {f(r['energy_mean'])}")
        for name, res in self.exams.items():
            lines.append(f"  last exam '{name}' at {res['t'] / 60:.1f} min: {summarize(name, res)}")
        stats = self.teacher.stats()
        if stats:
            lines.append(f"  teacher: {stats}")
        return "\n".join(lines)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "target": self.target, "reviewing": self.reviewing,
                "review_until": self.review_until, "curriculum": self.curriculum.state_dict(),
                "meter": self.meter.state_dict(), "toolkit": self.toolkit.state_dict(),
                "exams": self.exams, "exam_log": list(self.exam_log), "history": list(self.history),
                "milestones": self.milestones}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        from .curriculum import ALL_LESSONS
        self.target = s.get("target", self.target)
        self.meter.load_state_dict(s.get("meter", {}))
        names = s.get("curriculum", {}).get("lessons")
        if names and names != [x.name for x in self.curriculum.lessons] and all(n in ALL_LESSONS for n in names):
            self.curriculum = Curriculum(tuple(ALL_LESSONS[n] for n in names))  # as it was saved
            self.teacher.bind(self.curriculum.lessons, n_neurons=int(self.agent.brain.net.N))
        self.curriculum.load_state_dict(s.get("curriculum", {}))
        self.toolkit.load_state_dict(s.get("toolkit", {}))
        self.reviewing = s.get("reviewing")
        self.review_until = float(s.get("review_until", 0.0))
        if self.curriculum.finished:
            self.toolkit.clear_rules()
            self.toolkit.set_conditions(None, True)
        elif "conditions" not in s.get("toolkit", {}):
            lesson = self.reviewing or self.curriculum.current.name
            self._apply_lesson(lesson, announce=False)
        self.history.extend(s.get("history", []))
        self.milestones = list(s.get("milestones", []))
        self.exams = dict(s.get("exams", {}))
        self.exam_log.extend(s.get("exam_log", []))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), indent=1, default=float), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def detach(self) -> None:
        """Stop teaching and give the creature back its normal world."""
        self.stop()
        self.toolkit.set_conditions(None, True)
        self.meter.detach()
        self.toolkit.detach()
        self.agent.remove_tick_hook(self._on_tick)


def tutor_path(brain_path: str | Path) -> Path:
    p = Path(brain_path)
    return p.with_name(p.name.removesuffix(".npz") + ".tutor.json")
