"""A local LLM raises the creature (via Ollama).

    ollama pull llama3.2                 # once (any chat model works)
    python examples/04_llm_tutor.py      # or: python examples/04_llm_tutor.py qwen3:4b

The language model is the teacher, not the brain: every few seconds it reads
a short report on the creature and decides what to do — talk to it, smile,
give food, switch reflex rules, review a lesson.  Fast reflexes deliver the
rewards within half a second, and all learning happens in the spiking
network through its own dopamine.  Lessons are graduated by measurement.

If Ollama is not running, a scripted teacher takes over so the example still
runs.  Progress is saved (tutored_brain.npz + tutored_brain.tutor.json):
run the script again to continue where the lessons left off.
"""

import sys
from pathlib import Path

from anima import Agent
from anima.tutor import OllamaClient, OllamaError, OllamaTeacher, ScriptedTeacher, Tutor, tutor_path

brain = Path("tutored_brain.npz")
agent = Agent.load(brain) if brain.exists() else Agent.create(seed=3)

model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"
client = OllamaClient(model=model)
try:
    print(f"Ollama {client.version()} at {client.host}")
    if not client.has_model():
        raise OllamaError(f"model {model!r} is not pulled (ollama pull {model})")
    teacher = OllamaTeacher(client)  # falls back to the scripted teacher on any failure
except OllamaError as exc:
    print(f"no LLM teacher: {exc}\n-> using the scripted teacher instead")
    teacher = ScriptedTeacher()

tutor = Tutor(agent, teacher)
if tutor_path(brain).exists():
    tutor.load(tutor_path(brain))

# 20 minutes of the creature's life, in real time.  It never waits for the
# teacher: while the model thinks, the creature keeps living.
try:
    print(tutor.train(minutes=20, speed=1.0))
finally:
    agent.save(brain)
    tutor.save(tutor_path(brain))
    print(f"saved {brain}; run again to continue, or meet it: anima chat --brain {brain}")
