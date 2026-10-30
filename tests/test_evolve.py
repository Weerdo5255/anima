"""Genes, the end-of-day decision, generations, archiving or deleting, saving and resuming."""

import copy
import json
import math

import numpy as np
import pytest

from anima import Agent
from anima.evolve import (GENES, Colony, ColonyConfig, crossover, decide, decide_improved, mutate, read_genes,
                          write_genes)


def _cfg(**kw):
    base = dict(population=3, seed=1, cortex=200, day_length_s=60.0)
    base.update(kw)
    return ColonyConfig(**base)


def test_genes_roundtrip_into_a_living_creature():
    a = Agent.create(seed=3, cortex_size=200, kin=True)
    st = a.state_dict()
    g = read_genes(st)
    assert g == {x.name: x.default for x in GENES}
    new = dict(g, explore=5.0, move_lr=0.1, sociability=3.0, eagerness=0.2, loneliness=1e6)
    b = Agent.from_state(write_genes(copy.deepcopy(st), new))
    got = read_genes(b.state_dict())
    assert got["explore"] == 5.0 and got["sociability"] == 3.0 and got["loneliness"] == 400.0  # clipped
    net = b.brain.net
    assert net.noise_sigma[net.populations["motor"].slice].max() == 5.0
    assert net.projections["sense.kin->motor"].rule.lr == 0.1
    assert b.brain.effects.social_reward == 3.0 and b.body.motor.cfg.threshold == 0.2
    assert np.array_equal(net.W, a.brain.net.W)  # genes are not synapses


def test_mutation_and_crossover():
    rng = np.random.default_rng(0)
    g = {x.name: x.default for x in GENES}
    for _ in range(50):
        m, changed = mutate(g, rng, sigma=0.5, rate=0.3)
        assert changed and all(m[c] != g[c] or m[c] in (x.low, x.high) for c in changed for x in GENES if x.name == c)
        assert all(x.low <= m[x.name] <= x.high for x in GENES)
        assert all(m[k] == g[k] for k in g if k not in changed)
    h = {k: v * 2 for k, v in g.items()}
    kids = [crossover(g, h, rng) for _ in range(20)]
    assert any(k["explore"] == g["explore"] for k in kids) and any(k["explore"] == h["explore"] for k in kids)


def test_who_stands_out():
    assert decide({"a": 3.0, "b": 1.0, "c": 1.5}, 0.25, 0.5)[0] == "a"
    assert decide({"a": 1.6, "b": 1.5, "c": 1.4}, 0.25, 0.5)[0] is None          # not 25% more
    assert decide({"a": 0.5, "b": 0.25, "c": 0.0}, 0.25, 0.5)[0] is None          # not 0.5 more
    assert decide({"a": 0.75, "b": 0.0, "c": 0.0}, 0.25, 0.5)[0] == "a"           # the others ate nothing
    assert decide({"a": 2.0, "b": 2.0, "c": 0.0}, 0.25, 0.5)[0] is None           # a tie
    assert decide({"a": 0.0, "b": 0.0}, 0.25, 0.5)[0] is None


def test_who_improved_most():
    prev = {"a": 1.0, "b": 2.0, "c": 1.0}
    assert decide_improved({"a": 2.0, "b": 2.1, "c": 1.2}, prev, 0.25, 0.5)[0] == "a"
    assert decide_improved({"a": 1.2, "b": 2.1, "c": 1.0}, prev, 0.25, 0.5)[0] is None   # nobody improved enough
    assert decide_improved({"a": 0.5, "b": 2.6, "c": 1.0}, prev, 0.25, 0.5)[0] == "b"
    assert decide_improved({"x": 3.0}, prev, 0.25, 0.5)[0] is None                      # no previous day


def test_selection_by_improvement_waits_for_a_second_day(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(compare="itself"), printer=None)
    col._outcomes(0, [("ate", {"energy": 1.0})])
    assert not col.end_of_day()["decision"]["stands_out"]          # day 1: nothing to compare with
    col._outcomes(1, [("ate", {"energy": 1.0})])
    rep = col.end_of_day()
    assert rep["decision"]["stands_out"] and rep["decision"]["best"] == col.members[1]
    assert "improved most" in col.day_report(rep)
    col.close()


def test_a_colony_lives_deterministically(tmp_path):
    runs = []
    for k in range(2):
        col = Colony.create(tmp_path / f"c{k}", _cfg(), printer=None)
        col.run(seconds=5.0)
        runs.append([(round(b["x"], 6), round(b["y"], 6)) for b in col.arena.bodies]
                    + [col.agent(i).brain.net.spike_count.sum() for i in range(3)])
        col.close()
    assert runs[0] == runs[1]


def test_the_best_and_its_variations_take_over(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(), printer=None)
    col.run(seconds=3.0)
    champ = col.members[1]
    col._outcomes(1, [("ate", {"energy": 1.0})])  # it ate four meals' worth
    w_parent = col.agent(1).brain.net.W.copy()
    genes_parent = read_genes(col.agent(1).state_dict())
    old = [c for c in col.members if c != champ]
    home = (col.arena.bodies[1]["x"], col.arena.bodies[1]["y"])
    rep = col.end_of_day()
    assert rep["decision"]["stands_out"] and rep["next"]["champion"] == champ
    assert col.members[1] == champ and set(col.members) & set(old) == set()
    for slot in (0, 2):  # children are born next to their parent
        b = col.arena.bodies[slot]
        assert math.hypot(b["x"] - home[0], b["y"] - home[1]) <= 2.0 + 1e-9
    for slot, cid in enumerate(col.members):
        rec = col.records[cid]
        if cid == champ:
            continue
        assert rec["parent"] == champ and rec["generation"] == 1 and rec["born_day"] == 2
        child = col.agent(slot)
        assert read_genes(child.state_dict()) != genes_parent
        W = child.brain.net.W
        assert not np.array_equal(W, w_parent) and np.corrcoef(W.ravel(), w_parent.ravel())[0, 1] > 0.99
        assert child.t_ms == col.agent(1).t_ms                      # same moment of the same day
        assert child.brain.net.rng.random() != col.agent(1).brain.net.rng.random()  # its own noise
    for cid in old:
        assert (tmp_path / "col" / "retired" / f"{cid}.npz").exists()
        assert not (tmp_path / "col" / f"{cid}.npz").exists()
        assert col.records[cid]["fate"] == "retired"
    assert col.day == 2
    col.close()


def test_delete_old_deletes_the_replaced(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(delete_old=True), printer=None)
    col.run(seconds=2.0)
    col._outcomes(0, [("ate", {"energy": 1.0})])
    old = col.members[1:]
    col.end_of_day()
    d = tmp_path / "col"
    for cid in old:
        assert not (d / f"{cid}.npz").exists() and not (d / f"{cid}.thoughts.jsonl").exists()
        assert not (d / "retired" / f"{cid}.npz").exists()
        assert col.records[cid]["files"] == "deleted"
    col.close()


def test_nobody_stands_out_so_everyone_stays(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(), printer=None)
    before = list(col.members)
    rep = col.end_of_day()
    assert not rep["decision"]["stands_out"] and col.members == before and col.day == 2
    col.close()


def test_random_selection_is_a_control(tmp_path):
    parents = set()
    for seed in range(1, 9):
        col = Colony.create(tmp_path / f"c{seed}", _cfg(selection="random", seed=seed), printer=None)
        best = col.members[0]
        col._outcomes(0, [("ate", {"energy": 1.0})])  # creature 0 stands out...
        rep = col.end_of_day()
        assert rep["decision"]["stands_out"] and rep["decision"]["best"] == best
        parents.add(rep["next"]["champion"] == best)  # ...but the parent is anyone
        assert rep["decision"]["parent"] == rep["next"]["champion"]
        col.close()
    assert parents == {True, False}
    with pytest.raises(ValueError):
        Colony.create(tmp_path / "bad", _cfg(selection="fittest"), printer=None)


def test_fresh_brains_with_inherit_genes(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(inherit="genes"), printer=None)
    col.run(seconds=2.0)
    col._outcomes(0, [("ate", {"energy": 1.0})])
    col.end_of_day()
    child = col.agent(1)
    assert not np.array_equal(child.brain.net.M, col.agent(0).brain.net.M)  # new wiring
    assert child.t_ms == 0.0 and col.records[col.members[1]]["origin"] == "new brain"
    assert abs(child.brain.drives.phase - col.agent(0).brain.drives.phase) < 1e-3  # same time of day
    col.close()


def test_save_resume_and_policy_overrides(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(), printer=None)
    col.run(seconds=2.0)
    col._outcomes(2, [("ate", {"energy": 1.0})])
    col.end_of_day()
    col.run(seconds=1.0)
    col.save()
    t, members, serial = col.t, list(col.members), col.serial
    pos = [(b["x"], b["y"]) for b in col.arena.bodies]
    col.close()
    again = Colony.load(tmp_path / "col", printer=None, margin=0.5, delete_old=True)
    assert (again.t, again.members, again.serial, again.day) == (t, members, serial, 2)
    assert [(b["x"], b["y"]) for b in again.arena.bodies] == pos
    assert again.cfg.margin == 0.5 and again.cfg.delete_old and again.cfg.population == 3
    with pytest.raises(ValueError, match="fixed"):
        Colony.load(tmp_path / "col", printer=None, population=5)
    again.run(seconds=1.0)
    data = json.loads((tmp_path / "col" / "colony.json").read_text())
    assert data["records"][members[2]]["days"][0]["fitness"] >= 1.0
    again.close()
    lone = Agent.load(tmp_path / "col" / f"{members[0]}.npz")  # a colony creature can live alone
    lone.run(0.5)
    assert lone.world.name == "foraging"


def test_a_colony_from_an_existing_creature(tmp_path):
    src = tmp_path / "pet.npz"
    pet = Agent.create(seed=5, cortex_size=200, day_length_s=90.0)
    pet.run(1.0)
    pet.save(src)
    before = src.read_bytes()
    col = Colony.create(tmp_path / "col", _cfg(), founder=src, printer=None)
    assert src.read_bytes() == before                       # the original is left alone
    assert col.cfg.day_length_s == 90.0
    assert np.array_equal(col.agent(0).brain.net.W, pet.brain.net.W)
    assert all(col.records[c]["parent"] == col.members[0] for c in col.members[1:])
    assert "sense.kin" not in col.agent(0).brain.inputs      # it never had a kin sense
    col.run(seconds=1.0)
    col.close()


def test_creatures_hear_each_other_by_name(tmp_path):
    col = Colony.create(tmp_path / "col", _cfg(population=2), printer=None)
    a = col.arena
    a.bodies[0].update(x=5.0, y=5.0)
    a.bodies[1].update(x=6.0, y=5.0)
    heard = []
    col.agent(1).on("heard", heard.append)
    col._spoke(0, "ba bo")
    col.tick()
    assert heard and heard[0]["from"] == col.names[0] and heard[0]["loudness"] > 0.8
    a.bodies[1].update(x=17.0, y=17.0)
    heard.clear()
    col._spoke(0, "ba")
    col.tick()
    assert not heard
    col.close()


def test_worker_processes_give_the_same_life(tmp_path):
    """Creatures in two worker processes live exactly as in one process."""
    out = []
    for workers in (1, 2):
        col = Colony.create(tmp_path / f"w{workers}", _cfg(), printer=None, workers=workers)
        col.run(seconds=3.0)
        col._outcomes(2, [("ate", {"energy": 1.0})])
        col.end_of_day()  # a generation turns over across processes too
        col.run(seconds=1.0)
        out.append(([(round(b["x"], 9), round(b["y"], 9), round(b["heading"], 9)) for b in col.arena.bodies],
                    [round(e, 9) for e in col.energy], col.members, dict(col.arena.counts)))
        col.close()
    assert out[0] == out[1]


def test_cli_arena_runs_resumes_and_replaces(tmp_path, capsys):
    from anima.cli import main
    d = tmp_path / "col"
    common = ["arena", "--dir", str(d), "--population", "3", "--cortex", "200", "--day", "40",
              "--report-every", "0", "--workers", "1"]
    main(common + ["--days", "1", "--thoughts", "all"])
    out = capsys.readouterr().out
    assert "a new colony" in out and "=== end of day 1 ===" in out and (d / "colony.json").exists()
    main(["arena", "--dir", str(d), "--status"])
    out = capsys.readouterr().out
    assert "family tree" in out and "Ada (c1" in out
    main(common + ["--new"] + ["--days", "0"])  # the old colony is moved aside
    out = capsys.readouterr().out
    assert "moved the old colony" in out and (tmp_path / "col.old1" / "colony.json").exists()
    main(common + ["--new", "--delete-old", "--days", "0"])  # ...or deleted
    out = capsys.readouterr().out
    assert "deleted the old colony" in out and not (tmp_path / "col.old2").exists()


def test_a_failing_worker_reports_its_error():
    from anima.evolve.hosting import RemoteHost, WorkerError
    h = RemoteHost()
    try:
        with pytest.raises(WorkerError, match="KeyError"):
            h.add(0, {"spec": {}}, "Ada", {"type": "foraging", "kwargs": {}})
        assert h.proc.is_alive()  # it survives a bad request
    finally:
        h.close()
    assert not h.proc.is_alive()
