"""The thought log: what gets in, what it says, where it goes."""

import json

from anima import Agent
from anima.mind import KINDS, ThoughtStream, format_thought, read_thoughts, thoughts_path
from anima.tools import FunctionTool, ToolBelt
from anima.tutor import ScriptedTeacher, Tutor


def _agent(**kw):
    return Agent.create(seed=3, cortex_size=300, **kw)


def test_thoughts_path_sits_next_to_the_brain(tmp_path):
    assert thoughts_path(tmp_path / "pet.npz") == tmp_path / "pet.thoughts.jsonl"
    assert thoughts_path("anima_brain.npz").name == "anima_brain.thoughts.jsonl"


def test_a_living_creature_thinks_and_the_log_is_written(tmp_path):
    a = _agent()
    events = []
    a.on("thought", events.append)
    ts = ThoughtStream(a, tmp_path / "t.jsonl")
    a.run(2.0)
    a.hear("ba ba :)")
    a.run(25.0)
    ts.close()
    st = ts.stats()
    assert st["total"] >= 5
    assert set(st["counts"]) <= set(KINDS)
    texts = [t["text"] for t in ts.recent(0)]
    assert 'someone says "ba ba"' in texts and "someone smiles at me" in texts
    # every line is JSON with the numbers it came from
    rows = read_thoughts(tmp_path / "t.jsonl")
    assert len(rows) == st["total"] and all({"t", "clock", "kind", "text", "data", "salience"} <= set(r) for r in rows)
    assert read_thoughts(tmp_path / "t.jsonl", n=2, kinds=("social",))[-1]["kind"] == "social"
    # the thought events carry their kind as "type"
    assert events and events[0]["kind"] == "thought" and events[0]["type"] in KINDS
    line = format_thought(rows[0])
    assert rows[0]["text"] in line and rows[0]["clock"] in line
    assert "\033[" in format_thought(rows[0], color=True)


def test_the_most_salient_candidate_wins_and_the_rest_fade():
    a = _agent(world="void")
    ts = ThoughtStream(a, None, blink_s=2.0, fade_s=1.0)
    ts.offer("a", "perception", "weak", salience=0.3)
    ts.offer("b", "evaluation", "strong", salience=0.9)
    ts._admit(a.t_ms / 1000.0)
    assert [t["text"] for t in ts.recent(0)] == ["strong"]
    # the workspace is closed for a moment, and "weak" fades meanwhile
    a.run(1.5)
    ts._admit(a.t_ms / 1000.0)
    assert "weak" not in [t["text"] for t in ts.recent(0)]
    assert ts.unconscious >= 1
    ts.close()


def test_habituation_lets_a_new_kind_of_thought_through():
    a = _agent(world="void")
    ts = ThoughtStream(a, None, blink_s=0.0)
    now = a.t_ms / 1000.0
    ts.offer("speech", "speech", "I say ba", salience=0.5)
    ts._admit(now)
    ts.offer("speech", "speech", "I say bo", salience=0.5)  # same family, just thought of
    ts.offer("wall", "perception", "a wall", salience=0.4)
    ts._admit(now + 0.1)
    assert ts.recent(1)[0]["text"] == "a wall"
    ts.close()


def test_rewards_are_evaluated_and_attributed():
    a = _agent()
    ts = ThoughtStream(a, None)
    a.run(1.0)
    a.call_soon(lambda ag: ag.brain.deliver_reward(0.9, "program:snack"))
    a.run(3.0)  # it gets in when the workspace next opens
    evals = ts.recent(0, kinds=("evaluation",))
    assert evals, ts.recent(0)
    assert evals[0]["text"].startswith("snack: better than I expected")
    assert evals[0]["data"]["cause"] == "program:snack" and evals[0]["data"]["phasic_dopamine"] > 0.2
    ts.close()


def test_program_use_reads_as_one_thought_with_its_answer():
    a = Agent.create(seed=3, cortex_size=300, hands=4)
    belt = ToolBelt(a, [FunctionTool(name="snack", hand=2, inline=True, latency_s=0.2,
                                     fn=lambda c: {"reward": 0.5, "say": "nyam"})])
    ts = ThoughtStream(a, None, tool_names=belt.hand_names())
    a.call_soon(lambda ag: belt._on_hand({"hand": 2}))
    a.run(1.0)
    prog = ts.recent(0, kinds=("program",))
    assert [t["text"] for t in prog] == ["I use snack (with hand 2) → 'nyam', reward +0.50"]
    assert prog[0]["data"]["reward"] == 0.5
    belt.close()
    ts.close()


def test_sleep_brings_dreams_of_stored_moments():
    a = _agent()
    ts = ThoughtStream(a, None)
    for k in range(6):  # a few rewarded moments get stored by the hippocampus
        a.call_soon(lambda ag: ag.brain.deliver_reward(0.8, "food"))
        a.run(1.0)
    assert a.brain.hippocampus.episodes
    a.call_soon(lambda ag: ag.brain.sleep.force("nrem", ag.brain.time_s))
    a.run(30.0)
    kinds = [t["kind"] for t in ts.recent(0)]
    assert "state" in kinds and "dream" in kinds
    dream = ts.recent(0, kinds=("dream",))[0]
    assert dream["text"].startswith("I dream of") and "felt" in dream["text"]
    ts.close()


def test_reflection_summarises_recent_memories():
    a = _agent()
    ts = ThoughtStream(a, None, reflect_every_s=5.0)
    for _ in range(8):
        a.call_soon(lambda ag: ag.brain.deliver_reward(0.7, "food"))
        a.run(1.0)
    refl = ts.recent(0, kinds=("reflection",))
    assert refl and refl[0]["text"].startswith("looking back")
    assert refl[0]["data"]["memories"] >= 1 and refl[0]["data"]["items"]
    ts.close()


def test_closing_detaches_and_flushes(tmp_path):
    a = _agent()
    ts = ThoughtStream(a, tmp_path / "x.jsonl")
    a.hear("ma")
    a.run(1.0)
    ts.close()
    n = len(read_thoughts(tmp_path / "x.jsonl"))
    a.hear("ma")
    a.run(1.0)
    assert len(read_thoughts(tmp_path / "x.jsonl")) == n >= 1
    for line in (tmp_path / "x.jsonl").read_text().splitlines():
        json.loads(line)


def test_the_teacher_sees_its_thoughts():
    a = _agent()
    ts = ThoughtStream(a, None)
    tutor = Tutor(a, ScriptedTeacher(), printer=None, mind=ts)
    a.run(1.0)
    a.call_soon(lambda ag: ag.brain.deliver_reward(0.9, "food"))
    a.run(3.0)
    obs = tutor.observe()
    assert "its_thoughts" in obs and all(isinstance(x, str) for x in obs["its_thoughts"])
    assert obs["its_thoughts"]
    assert "its_thoughts" not in Tutor(_agent(), ScriptedTeacher(), printer=None).observe()
    tutor.stop()
    ts.close()


def test_gestures_are_not_reported_as_words():
    a = _agent(world="void")
    ts = ThoughtStream(a, None)
    a.hear("*smile*")
    a.hear("ma ma :)")
    a.run(0.1)
    texts = [t["text"] for t in ts.recent(0, kinds=("social",))]
    assert texts == ["someone smiles at me", 'someone says "ma ma"', "someone smiles at me"]
    ts.close()
