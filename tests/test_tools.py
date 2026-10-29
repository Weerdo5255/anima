"""Programs the creature can use: triggers, kinds, answers, limits, hands."""

import sys

import numpy as np
import pytest

from anima import Agent
from anima.brain.presets import infant_spec
from anima.tools import (CommandTool, FunctionTool, RemoteTool, ToolBelt, load_tools, parse_answer,
                         tool_from_dict)


def _hands_agent(**kw):
    return Agent.create(seed=3, hands=4, cortex_size=300, **kw)


def _results(agent):
    out = []
    agent.on("tool_result", out.append)
    return out


def test_parse_answer_text_json_and_dict():
    r = parse_answer("  hello   there \n", reward_ok=0.2)
    assert (r.ok, r.say, r.reward) == (True, "hello there", 0.2)
    r = parse_answer('{"say": "ding", "reward": 0.5, "food": "left"}')
    assert (r.say, r.reward, r.food) == ("ding", 0.5, "left")
    r = parse_answer({"food": True, "ok": False})
    assert r.food == "ahead" and not r.ok
    r = parse_answer('{"food": "upstairs", "reward": "lots"}')
    assert r.food is None and r.reward == 0.0
    assert parse_answer("{not json").say == "{not json"
    assert parse_answer(b"bytes ok").say == "bytes ok"


def test_hands_are_opt_in_and_leave_the_default_brain_unchanged():
    assert infant_spec(seed=3) == infant_spec(seed=3, hands=0)
    names = {p["name"] for p in infant_spec(seed=3, hands=4)["population"]}
    assert {"hands", "hands.inh", "sense.hands"} <= names
    a = _hands_agent()
    assert a.body.hands is not None and len(a.body.hands.labels) == 4
    assert Agent.create(seed=3, cortex_size=300).body.hands is None


def test_hand_program_runs_and_its_answer_comes_back_as_consequences():
    a = _hands_agent()
    seen = []
    belt = ToolBelt(a, [FunctionTool(name="snack", hand=1, inline=True, latency_s=0.2,
                                     fn=lambda c: (seen.append(c), {"reward": 0.6, "say": "nyam"})[1])])
    res = _results(a)
    a.run(0.1)
    a.call_soon(lambda ag: belt._on_hand({"hand": 1}))
    a.call_soon(lambda ag: belt._on_hand({"hand": 2}))  # nothing bound: an empty reach
    a.run(0.5)
    assert len(seen) == 1 and seen[0]["how"] == "hand" and seen[0]["tool"] == "snack"
    assert set(seen[0]["drives"]) == {"hunger", "fatigue", "boredom", "loneliness"}
    assert len(res) == 1 and res[0]["reward"] == 0.6 and res[0]["say"] == "nyam"
    assert any(src == "program:snack" and amt == 0.6 for _, amt, src in a.brain.reward_log)
    assert belt.stats["snack"]["ok"] == 1 and belt.empty_hands == 1
    assert belt.hand_names() == {1: "snack"}
    belt.close()


def test_say_triggers_prefix_and_exact_syllables():
    a = Agent.create(seed=3, cortex_size=300, world="void")
    calls = []
    belt = ToolBelt(a, [FunctionTool(name="bee", say="b", inline=True, latency_s=0, cooldown_s=0, fn=lambda c: calls.append(c["what"])),
                        FunctionTool(name="la", say="la", inline=True, latency_s=0, cooldown_s=0, fn=lambda c: calls.append(c["what"]))])
    for syl in ("ba", "bo", "ma", "la", "lo"):
        belt._on_syllable({"text": syl})
        a.run(0.01)
    assert calls == ["ba", "bo", "la"]
    belt.close()


def test_cooldown_and_rate_limit_skip_calls():
    a = Agent.create(seed=3, cortex_size=300, world="void")
    belt = ToolBelt(a, [FunctionTool(name="t", say="b", inline=True, cooldown_s=5.0, fn=lambda c: "ok")],
                    max_calls_per_min=30)
    belt._on_syllable({"text": "ba"})
    belt._on_syllable({"text": "bi"})
    assert belt.stats["t"]["calls"] == 1 and belt.stats["t"]["skipped"] == 1
    belt.close()
    belt = ToolBelt(a, [FunctionTool(name="t", say="b", inline=True, latency_s=0.0, cooldown_s=0.0,
                                     fn=lambda c: "ok")], max_calls_per_min=2)
    for _ in range(4):
        belt._on_syllable({"text": "ba"})
        a.run(0.01)
    assert belt.stats["t"]["calls"] == 2 and belt.stats["t"]["skipped"] == 2
    belt.close()


def test_bindings_are_validated():
    a = Agent.create(seed=3, cortex_size=300, world="void")
    with pytest.raises(ValueError, match="no hands"):
        ToolBelt(a, [FunctionTool(name="x", hand=0, fn=lambda c: "")])
    with pytest.raises(ValueError, match="exactly one"):
        ToolBelt(a, [FunctionTool(name="x", fn=lambda c: "")])
    with pytest.raises(ValueError, match="letters"):
        ToolBelt(a, [FunctionTool(name="x", say="b!", fn=lambda c: "")])
    with pytest.raises(ValueError, match="unique"):
        ToolBelt(a, [FunctionTool(name="x", say="b", fn=lambda c: ""), FunctionTool(name="x", say="m", fn=lambda c: "")])
    h = _hands_agent()
    with pytest.raises(ValueError, match="does not exist"):
        ToolBelt(h, [FunctionTool(name="x", hand=7, fn=lambda c: "")])


def test_command_tool_fills_placeholders_reads_stdin_and_never_uses_a_shell(tmp_path):
    code = ("import sys, json; c = json.load(sys.stdin); "
            "print(json.dumps({'say': sys.argv[1] + ' ' + sys.argv[2] + ' ' + c['tool'], 'reward': 3}))")
    t = CommandTool(name="probe", say="b", argv=["{python}", "-c", code, "{hunger}", "{literal}; echo hi"])
    call = {"call_id": 1, "tool": "probe", "what": "ba", "t": 1.0, "drives": {"hunger": 0.25}}
    r = t.run(call)
    assert r.ok and r.say == "0.250 {literal}; echo hi probe"
    assert r.reward == 1.0  # clipped to [-1, 1]
    slow = CommandTool(name="slow", say="b", argv=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=0.3)
    r = slow.run(call)
    assert not r.ok and "timed out" in r.error
    bad = CommandTool(name="bad", say="b", argv=[sys.executable, "-c", "import sys; sys.exit(3)"], reward_error=-0.2)
    r = bad.run(call)
    assert not r.ok and r.reward == -0.2 and "exit 3" in r.error
    missing = CommandTool(name="missing", say="b", argv=[str(tmp_path / "no_such_program")])
    assert not missing.run(call).ok
    with pytest.raises(ValueError, match="list"):
        tool_from_dict({"name": "x", "say": "b", "command": "echo hi"})


def test_command_programs_run_in_the_background(tmp_path):
    a = Agent.create(seed=3, cortex_size=300, world="void")
    res = _results(a)
    belt = ToolBelt(a, [CommandTool(name="echo", say="b", argv=["{python}", "-c", "print('hello')"])])
    belt._on_syllable({"text": "ba"})
    for _ in range(300):
        a.run(0.05)
        if res:
            break
    assert res and res[0]["say"] == "hello" and res[0]["ok"]
    belt.close()


def test_remote_program_is_answered_from_outside_or_times_out():
    a = Agent.create(seed=3, cortex_size=300, world="void")
    res = _results(a)
    calls = []
    a.on("tool_call", calls.append)
    belt = ToolBelt(a, [RemoteTool(name="bell", say="b", timeout_s=1.0, cooldown_s=0.0)])
    belt._on_syllable({"text": "ba"})
    pending = belt.pending_remote()
    assert len(pending) == 1 and calls[0]["program"] == "remote" and calls[0]["call"]["tool"] == "bell"
    cid = pending[0]["call_id"]
    assert belt.complete(cid, {"say": "ding", "reward": 5})
    assert not belt.complete(cid, "again")  # already answered
    a.run(0.05)
    assert res[-1]["say"] == "ding" and res[-1]["reward"] == 1.0
    belt._on_syllable({"text": "bo"})
    a.run(1.2)
    assert not res[-1]["ok"] and "no answer" in res[-1]["error"]
    assert belt.pending_remote() == []
    belt.close()


def test_food_answers_place_food():
    a = _hands_agent()
    belt = ToolBelt(a, [FunctionTool(name="vend", hand=0, inline=True, latency_s=0.0, fn=lambda c: {"food": "ahead"})])
    n0 = len(a.world.food)
    belt._on_hand({"hand": 0})
    a.run(0.05)
    assert len(a.world.food) == n0 + 1
    belt.close()


def test_load_tools_from_toml_and_json(tmp_path):
    (tmp_path / "t.toml").write_text('''
[[tool]]
name = "vending"
hand = 0
command = ["{python}", "vend.py"]
cooldown_s = 2

[[tool]]
name = "bell"
say = "B"
remote = true

[[tool]]
name = "web"
hand = 1
url = "http://127.0.0.1:9/x"
''')
    tools = load_tools(tmp_path / "t.toml")
    assert [t.kind for t in tools] == ["command", "remote", "http"]
    assert tools[0].cwd == str(tmp_path.resolve()) and tools[0].cooldown_s == 2
    assert tools[1].say == "b"
    (tmp_path / "t.json").write_text('{"tool": [{"name": "f", "say": "m", "function": "builtins:len"}]}')
    (f,) = load_tools(tmp_path / "t.json")
    assert f.kind == "function" and f.run({"x": 1}).say == "1"
    with pytest.raises(ValueError, match="exactly one"):
        tool_from_dict({"name": "x", "say": "b", "remote": True, "url": "http://x"})


def test_the_example_tools_file_loads():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tools = load_tools(root / "examples" / "tools.toml")
    assert {t.name for t in tools} == {"vending", "clock", "music"}
    music = [t for t in tools if t.name == "music"][0]
    r = music.run({"call_id": 1, "tool": "music", "drives": {"boredom": 0.8}})
    assert r.ok and r.reward == pytest.approx(0.4) and r.say
    (bell,) = load_tools(root / "examples" / "tools_remote.toml")
    assert bell.kind == "remote" and bell.say == "b"


def test_hands_creature_saves_and_loads_with_its_hands(tmp_path):
    a = _hands_agent()
    a.run(1.0)
    a.save(tmp_path / "h.npz")
    b = Agent.load(tmp_path / "h.npz")
    assert b.body.hands is not None and b.body.cfg.hands.threshold == a.body.cfg.hands.threshold
    assert np.array_equal(b.brain.net.W, a.brain.net.W)
