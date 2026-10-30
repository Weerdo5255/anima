"""The shared arena: shared food, feasts that need two, sensing and hearing each other."""

import json
import math

import numpy as np

from anima import Agent
from anima.world import ForagingWorld
from anima.world.arena import SharedArena


def _arena(n=2, **kw):
    a = SharedArena(seed=1, n_food=0, n_feasts=0, **kw)
    for k in range(n):
        a.add(f"C{k}")
    return a


def _put(a, i, x, y, heading=0.0):
    a.bodies[i].update(x=x, y=y, heading=heading)


def test_food_is_shared_and_the_rival_nearby_misses_it():
    a = _arena(2)
    a.food = [[5.7, 5.0]]
    _put(a, 0, 5.0, 5.0, 0.0)
    _put(a, 1, 6.5, 6.0)
    ev = a.apply(0, "forward")
    assert ev[0][0] == "ate" and ev[0][1]["beat"] == ["C1"]
    assert a.food == [] and len(a.regrow) == 1
    assert a.take_pending(1) == [("missed", {"by": "C0"})]
    assert a.counts == {"meals": 1, "feasts": 0, "contested": 1, "withered": 0}
    a.step(13.0)  # it grows back somewhere
    assert len(a.food) == 1 and a.regrow == []


def test_a_feast_needs_two_and_feeds_both():
    a = _arena(3)
    a.feasts = [[10.0, 10.0]]
    _put(a, 0, 9.0, 10.0, 0.0)
    _put(a, 1, 10.0, 8.0, math.pi / 2)
    _put(a, 2, 2.0, 2.0)
    assert a.apply(0, "forward") == [("feast_alone", {})]   # alone: too big
    assert a.apply(0, "left") == []                          # still there: said once
    ev = a.apply(1, "forward")                               # the second one arrives
    assert ev[0][0] == "ate" and ev[0][1]["feast"] and ev[0][1]["with"] == ["C0"]
    other = a.take_pending(0)
    assert other[0][0] == "ate" and other[0][1]["with"] == ["C1"] and other[0][1]["energy"] == a.feast_energy
    assert a.take_pending(2) == [] and a.feasts == [] and a.counts["feasts"] == 1


def test_kin_code_points_at_the_others():
    a = _arena(2)
    _put(a, 0, 9.0, 9.0, 0.0)
    _put(a, 1, 9.0, 11.0)          # 90 degrees to the left of a creature facing +x
    kin = a.observe(0)["kin"]
    labels = [int(x) for x in range(-180, 180, 45)]
    assert labels[int(np.argmax(kin))] == 90
    assert _arena(1).observe(0)["kin"].sum() == 0.0


def test_a_view_is_a_world_and_saves_like_a_foraging_world():
    a = SharedArena(seed=2, n_food=4, n_feasts=1)
    v = a.view(a.add("Ada"))
    agent = Agent.create(seed=3, cortex_size=200, kin=True)
    agent = Agent(agent.brain, v, name="Ada", world_spec={"type": "foraging", "kwargs": {"size": a.size}})
    for _ in range(300):
        agent.tick()
        a.step(0.01)
    assert agent.brain.get_input("sense.kin").sum() == 0.0  # alone
    w = ForagingWorld(size=a.size)
    w.load_state_dict(v.state_dict())
    assert (w.x, w.y) == (v.body["x"], v.body["y"]) and len(w.food) == len(a.food)


def test_arena_state_roundtrip():
    a = SharedArena(seed=4)
    a.add("Ada")
    a.add("Bo")
    for _ in range(50):
        a.apply(0, "forward")
        a.apply(1, "left")
        a.step(0.5)
    b = SharedArena.from_state(a.state_dict())
    assert b.food == a.food and b.bodies[1]["heading"] == a.bodies[1]["heading"]
    assert b.observe(0)["bearing"].tolist() == a.observe(0)["bearing"].tolist()
    assert b.rng.random() == a.rng.random()


def test_food_nobody_eats_withers_and_grows_back_elsewhere():
    a = SharedArena(seed=3, n_food=2, n_feasts=1, wither_s=(5.0, 6.0))
    a.add("C0")
    _put(a, 0, 0.5, 0.5)
    before = [list(f) for f in a.food]
    for _ in range(4):
        a.step(1.5)  # 6 s: all withered, growing back
    assert a.food == [] and a.feasts == [] and a.counts["withered"] == 3
    assert len(a.regrow) == 2 and len(a.feast_regrow) == 1
    a.wither_s = (1000.0, 1000.0)  # what grows from now on lasts
    for _ in range(30):
        a.step(1.0)
    assert len(a.food) == 2 and a.food != before and len(a.feasts) == 1
    b = SharedArena(seed=3, n_food=2, wither_s=None)  # food that never withers
    for _ in range(100):
        b.step(10.0)
    assert len(b.food) == 2 and b.counts["withered"] == 0


def test_gifts_do_not_grow_back_and_outside_food_is_adopted():
    a = _arena(1)
    _put(a, 0, 5.0, 5.0, 0.0)
    a.place_food(0, 0.0, 0.6)  # right ahead
    assert a.gift == [True]
    ev = a.apply(0, "forward")
    assert ev[0][0] == "ate" and ev[0][1]["gift"] and a.regrow == []
    a.food.append([12.0, 12.0])  # someone put food straight into the list
    a.step(0.01)
    assert a.gift == [False] and len(a.food_life) == 1


def test_wither_state_roundtrip_and_old_saves():
    a = SharedArena(seed=4, n_food=3, n_feasts=1)
    a.add("C0")
    a.place_food(0, 0.0, 3.0)
    a.step(1.0)
    s = json.loads(json.dumps(a.state_dict()))
    b = SharedArena.from_state(s)
    assert b.food == a.food and b.food_life == a.food_life and b.gift == a.gift and b.feast_life == a.feast_life
    a.step(70.0)
    b.step(70.0)
    assert b.food == a.food and b.counts == a.counts
    for k in ("food_life", "gift", "feast_life"):  # a colony saved before food withered
        del s[k]
    del s["kwargs"]["wither_s"]
    old = SharedArena.from_state(s)
    assert old.wither_s is None and old.gift == [False] * len(old.food)
