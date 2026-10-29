"""Teaching an anima creature with a local LLM (Ollama) or a scripted teacher.

The language model never replaces the spiking brain and never writes its
weights.  It acts like a parent or animal trainer: it watches, decides what
to teach next, talks to the creature, rewards it and arranges its world —
while fast reflex rules deliver rewards within the half-second the
creature's synapses can link to its behaviour.  All learning still happens
through the creature's own local plasticity and dopamine.  Progress is
measured passively and by exams taken by a clone of the creature, and a
four-lesson curriculum takes it from helpless to independent.

    from anima import Agent
    from anima.tutor import Tutor, OllamaTeacher

    agent = Agent.create()
    tutor = Tutor(agent, OllamaTeacher(model="llama3.2"))
    print(tutor.train(minutes=30))
"""

from .curriculum import ASK, LESSON_NAMES, LESSONS, LESSONS_WITH_ASK, Curriculum, Lesson
from .exams import EXAMS, ask_exam, forage_exam
from .ollama import (DEFAULT_MODEL, OllamaClient, OllamaError, OllamaModelMissing, OllamaUnavailable,
                     extract_json)
from .skills import SkillMeter, SkillReport
from .teachers import DECISION_SCHEMA, Decision, OllamaTeacher, ScriptedTeacher, Teacher, validate
from .toolkit import RULES, TeachingToolkit
from .tutor import Tutor, tutor_path

__all__ = [
    "Tutor", "tutor_path", "Teacher", "ScriptedTeacher", "OllamaTeacher", "Decision", "validate",
    "DECISION_SCHEMA", "OllamaClient", "OllamaError", "OllamaUnavailable", "OllamaModelMissing",
    "extract_json", "DEFAULT_MODEL", "SkillMeter", "SkillReport", "TeachingToolkit", "RULES",
    "Curriculum", "Lesson", "LESSONS", "LESSON_NAMES", "ASK", "LESSONS_WITH_ASK", "EXAMS", "forage_exam",
    "ask_exam",
]
