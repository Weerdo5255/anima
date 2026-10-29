"""UI logic that can be tested without a terminal."""

from anima import Agent
from anima.ui.chat import ChatUI
from anima.ui.tui import Dashboard, _bar, _spark


def test_dashboard_collects_conversation_and_samples():
    agent = Agent.create(seed=3)
    dash = Dashboard(agent, None)
    agent.run(0.5)
    dash.handle_line("hello mama")
    agent.run(2)
    dash._drain()
    dash._sample()
    who = [w for w, _ in dash.chat]
    assert "you" in who
    assert dash.raster.shape[0] == 10
    assert all(len(v) == 1 for v in dash.rates.values())


def test_dashboard_commands():
    agent = Agent.create(seed=3, world="void")
    dash = Dashboard(agent, None)
    assert dash.handle_line("/sleep") is True
    agent.run(0.05)
    assert agent.brain.stage == "nrem"
    assert dash.handle_line("/quit") is False


def test_chat_commands(tmp_path):
    agent = Agent.create(seed=3, world="void")
    lines = []
    ui = ChatUI(agent, tmp_path / "c.npz", out=lines.append)
    assert ui.command("/status")
    assert ui.command("/save")
    assert (tmp_path / "c.npz").exists()
    assert ui.command("/nonsense")
    assert not ui.command("/quit")
    assert any("saved" in s for s in lines)


def test_bars_and_sparklines():
    assert _bar(0.5, 10) == "█████·····"
    assert len(_spark([1, 2, 3], 8)) == 8
    assert _spark([], 4) == "    "


def test_chat_shows_thoughts_and_programs(tmp_path):
    from anima.mind import ThoughtStream
    from anima.tools import FunctionTool, ToolBelt
    agent = Agent.create(seed=3, world="void")
    belt = ToolBelt(agent, [FunctionTool(name="bell", say="b", fn=lambda c: "ding")])
    mind = ThoughtStream(agent, None)
    lines = []
    ui = ChatUI(agent, None, out=lines.append, belt=belt, mind=mind)
    agent.hear("ma ma")
    agent.run(3.0)
    assert ui.command("/thoughts 5")
    assert any('someone says "ma ma"' in s for s in lines)
    assert ui.command("/mind") and ui.show_thoughts
    assert ui.command("/tools") and any("bell" in s for s in lines)
    bare = ChatUI(agent, None, out=lines.append)
    bare.command("/thoughts")
    bare.command("/tools")
    assert any("not being read" in s for s in lines) and any("no programs" in s for s in lines)
    belt.close()
    mind.close()


def test_dashboard_lists_thoughts():
    from anima.mind import ThoughtStream
    agent = Agent.create(seed=3)
    mind = ThoughtStream(agent, None)
    dash = Dashboard(agent, None, mind=mind)
    agent.run(3.0)
    dash._drain()
    assert any(who == "thought" for who, _ in dash.chat)
    dash.handle_line("/mind")
    assert not dash.show_thoughts
    mind.close()


def test_cli_tools_and_thoughts_commands(tmp_path, capsys):
    import json as _json
    from pathlib import Path

    from anima.cli import main
    root = Path(__file__).resolve().parents[1]
    main(["tools", str(root / "examples" / "tools.toml"), "--try", "music"])
    out = capsys.readouterr().out
    assert "3 program(s)" in out and "tried music" in out
    log = tmp_path / "pet.thoughts.jsonl"
    log.write_text("\n".join(_json.dumps({"t": i, "clock": "08:00", "kind": k, "text": f"thought {i}", "data": {}})
                             for i, k in enumerate(["feeling", "intention", "feeling"])) + "\n")
    main(["thoughts", str(tmp_path / "pet.npz"), "--kinds", "feeling"])
    out = capsys.readouterr().out
    assert "thought 0" in out and "thought 2" in out and "thought 1" not in out


def test_cli_run_with_programs_and_thoughts(tmp_path, capsys):
    from pathlib import Path

    from anima.cli import main
    root = Path(__file__).resolve().parents[1]
    brain = tmp_path / "pet.npz"
    main(["run", "--brain", str(brain), "--new", "--cortex", "300", "--seconds", "12",
          "--tools", str(root / "examples" / "tools.toml"), "--thoughts"])
    out = capsys.readouterr().out
    assert "programs:" in out and "thoughts:" in out and brain.exists()
    assert (tmp_path / "pet.thoughts.jsonl").exists()
    from anima import Agent
    assert len(Agent.load(brain).body.hands.labels) == 4  # hands added because tools.toml binds them
