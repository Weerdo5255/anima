"""The agent: autonomy, asynchronous input, persistence, worlds, voice."""

import threading
import time

import numpy as np
import pytest

from anima import Agent, ForagingWorld
from anima.body.voice import parse_social_signals, text_to_phonemes
from anima.brain.presets import CONSONANTS, VOWELS


@pytest.fixture(scope="module")
def lived():
    """One infant that lived 30 s with nobody there (shared by several tests)."""
    agent = Agent.create(seed=3, world="foraging")
    events = []
    agent.on("*", lambda e: events.append(e))
    agent.run(30)
    return agent, events


def test_acts_and_babbles_without_any_prompt(lived):
    agent, events = lived
    kinds = [e["kind"] for e in events]
    assert kinds.count("action") > 10, "it should move on its own"
    assert kinds.count("syllable") > 3, "it should babble on its own"
    assert "heard" not in kinds  # nobody said anything


def test_drives_changed_while_alone(lived):
    agent, _ = lived
    d = agent.latest["drives"]
    assert d["loneliness"] > 0.35  # nobody came


def test_hearing_is_asynchronous_and_relieves_loneliness():
    agent = Agent.create(seed=4, world="void")
    agent.run(5)
    before = agent.brain.drives.loneliness
    audio = agent.brain.net.populations["audio"]
    c0 = agent.brain.net.spike_count[audio.slice].sum()
    t = threading.Thread(target=lambda: agent.hear("hello little one"))
    t.start()
    t.join()
    agent.run(2)
    assert agent.brain.net.spike_count[audio.slice].sum() - c0 > 50  # it heard the sounds
    assert agent.brain.drives.loneliness < before
    assert any(k == "heard" for _, k, _ in agent.log)


def test_background_thread_runs_in_realtime():
    agent = Agent.create(seed=5, world="void")
    agent.start(speed=1.0)
    time.sleep(1.5)
    agent.hear("ba ba")  # does not block
    time.sleep(0.5)
    agent.stop()
    simulated = agent.t_ms / 1000.0
    assert 1.2 < simulated < 2.6, simulated


def test_save_and_load_continue_the_same_life(tmp_path):
    a = Agent.create(seed=6)
    a.run(3)
    path = a.save(tmp_path / "creature.npz")
    b = Agent.load(path)
    np.testing.assert_array_equal(a.brain.net.W, b.brain.net.W)
    assert b.t_ms == a.t_ms
    assert b.brain.drives.loneliness == a.brain.drives.loneliness
    a.run(1)
    b.run(1)
    np.testing.assert_array_equal(a.brain.net.spike_count, b.brain.net.spike_count)


def test_clone_of_a_living_creature_is_exact_and_independent():
    a = Agent.create(seed=8)
    a.start(speed=4.0)
    time.sleep(0.5)
    c = a.clone()  # snapshot taken on the creature's own thread, mid-life
    a.stop()
    assert c.t_ms <= a.t_ms
    lag = a.t_ms - c.t_ms
    c.run(lag / 1000.0)  # catch the clone up: it lives exactly the same life
    np.testing.assert_array_equal(a.brain.net.spike_count, c.brain.net.spike_count)
    c.brain.net.W[0, 1] += 1.0
    assert a.brain.net.W[0, 1] != c.brain.net.W[0, 1]


def test_text_to_phonemes():
    seq = [p for p, _ in text_to_phonemes("Hello, mama!", CONSONANTS, VOWELS) if p]
    assert seq == list("helloma ma".replace(" ", ""))
    assert [p for p, _ in text_to_phonemes("very", CONSONANTS, VOWELS) if p] == list("beli")


def test_social_signals():
    text, val = parse_social_signals("good job :) *smile*")
    assert val > 0 and ":)" not in text
    assert parse_social_signals("no :(")[1] < 0


def test_foraging_world_mechanics():
    w = ForagingWorld(seed=0, n_food=1)
    w.x, w.y, w.heading = 5.0, 5.0, 0.0
    w.food = [[7.0, 5.0]]
    obs = w.observe()
    assert obs["bearing"].argmax() == 4  # straight ahead (0 deg channel)
    w.apply("left")
    assert w.observe()["bearing"].argmax() < 4  # food now to the right
    w.apply("right")
    ev = w.apply("forward") + w.apply("forward")
    assert any(e[0] == "ate" for e in ev)
    w.x, w.heading = 15.6, 0.0
    assert any(e[0] == "bump" for e in w.apply("forward"))
