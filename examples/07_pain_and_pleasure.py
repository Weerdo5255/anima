"""Teach it how to react, with pleasure and pain given a little late.

    python examples/07_pain_and_pleasure.py          # about two minutes

You say a word ("ba ba", then "ki ki", ...).  When its reply repeats at
least the first sound you give it pleasure, otherwise a little pain — two
to four seconds after it fell silent, about as late as a person typing
/pleasure or /pain in the chat.  That is too late for its synapses on their
own (their traces last a second or two), but pleasure and pain make it
recall what it just said, so the credit lands on its reply.

Before and after, a copy of it (plasticity off) is examined: how often does
its reply repeat the examiner's word?  The same thing happens in `anima
chat` when you type /pleasure, /pain, + or - after its replies.
"""

import numpy as np

from anima import Agent
from anima.tutor.exams import imitate_exam, summarize
from anima.tutor.skills import WordTrial

agent = Agent.create(seed=4)
agent.run(seconds=60)  # a minute of babbling on its own first
print("before:", summarize("imitate", imitate_exam(agent, words=33, seed=1)))

rng = np.random.default_rng(0)
heard = []
agent.on("syllable", heard.append)
feedback = []  # (due time, kind, amount)
words = ["ba ba", "ki ki", "do do", "mu mu", "te te", "go go"]
word, wins, tries, given = words[0], 0, 0, {"pleasure": 0, "pain": 0}
while agent.t_ms < 60_000 + 4 * 60_000:  # four minutes of teaching
    if wins >= 2 or tries >= 8:  # move on once it has repeated the word twice (or after 8 tries)
        word = words[(words.index(word) + 1) % len(words)]
        wins, tries = 0, 0
    trial = WordTrial(word, agent.t_ms / 1000)
    agent.hear(word)
    tries += 1
    seen, was = len(heard), True
    while not trial.closed:
        agent.tick()
        t = agent.t_ms / 1000
        hearing = agent.body.ear.external_pending
        if was and not hearing:
            trial.heard_end(t)
        was = hearing
        for e in heard[seen:]:
            trial.syllable(e["t"], e["text"], e["act"])
        seen = len(heard)
        trial.tick(t)
        for f in [f for f in feedback if f[0] <= t]:  # late feedback, as from a person
            agent.feel(f[1], f[2], "auto")  # "auto": its last words, as /pleasure and /pain in the chat
            feedback.remove(f)
    res = trial.result
    if res["answered"]:
        good = res["level"] in ("consonant", "syllable", "word")
        wins += good
        kind = "pleasure" if good else "pain"
        given[kind] += 1
        delay = 1.0 + rng.uniform(1.5, 3.5)  # it falls silent, you read it, you type
        feedback.append((agent.t_ms / 1000 + delay, kind, 0.8 if good else 0.4))
    for _ in range(int(rng.uniform(300, 500))):  # a pause before the next word
        agent.tick()
        t = agent.t_ms / 1000
        for f in [f for f in feedback if f[0] <= t]:
            agent.feel(f[1], f[2], "auto")
            feedback.remove(f)

print(f"taught: {given['pleasure']} times pleasure, {given['pain']} times pain")
print("after: ", summarize("imitate", imitate_exam(agent, words=33, seed=1)))
