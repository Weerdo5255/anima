"""Leave the creature alone for two simulated minutes and watch what it does.

Nobody talks to it and nothing is required of it.  Everything printed below
is produced by its own dynamics: background noise + recurrent activity,
drives that build up, and competitions in its motor and vocal populations.

    python examples/01_alone.py
"""

from anima import Agent

agent = Agent.create(seed=3)


def show(e):
    if e["kind"] == "utterance":
        print(f"{e['t']:6.1f}s  says  {e['text']}")
    elif e["kind"] == "ate":
        print(f"{e['t']:6.1f}s  eats")
    elif e["kind"] == "stage":
        print(f"{e['t']:6.1f}s  -> {e['stage']}")


agent.on("*", show)
for _ in range(4):
    agent.run(30)  # as fast as the CPU allows; use realtime=True to pace it
    print("        " + agent.describe())
print("lifetime counters:", agent.body.stats)
