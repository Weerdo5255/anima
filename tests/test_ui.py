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
