"""Run the creature in a background thread and talk to it from your code.

The agent lives in real time on its own thread.  ``hear()`` never blocks and
never "asks" for a reply: the words become sound in its auditory cortex, and
whatever it does next is up to its dynamics.

    python examples/02_talk_in_background.py
"""

import time

from anima import Agent

agent = Agent.create(seed=5)
agent.on("utterance", lambda e: print(f"   anima: {e['text']}"))
agent.on("stage", lambda e: print(f"   (now {e['stage']})"))
agent.start(speed=1.0)  # real time

script = [(3, "hello"), (6, "mama"), (9, "mama"), (13, ":)"), (18, "baba"), (24, "good :)")]
t0 = time.time()
for at, words in script:
    time.sleep(max(0.0, at - (time.time() - t0)))
    print(f"you: {words}")
    agent.hear(words)
time.sleep(6)
agent.stop()
print(agent.describe())
agent.save("talked_to.npz")
print("saved talked_to.npz — Agent.load('talked_to.npz') continues this life")
