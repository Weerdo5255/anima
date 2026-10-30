"""The program and colony experiments' plumbing and metrics (tiny runs; the real ones take hours)."""

import json

import math

from anima.experiments import colony as C
from anima.experiments import programs as P


def test_metrics_on_handmade_data():
    actions = [(t * 0.4, ["left", "left", "right", "forward"][int(t) % 4]) for t in range(200)]
    thoughts = [{"t": 10.0, "kind": "intention", "data": {"move": "left"}},
                {"t": 20.0, "kind": "intention", "data": {"move": "right"}}]
    r = P.intention_check(thoughts, actions)
    assert r["n"] == 2 and 0.0 <= r["came_true"] <= 1.0 and 0.0 < r["chance"] < 1.0
    reaches = [(float(t), 1 if t < 50 else 0) for t in range(0, 100, 5)]
    urges = [{"t": 47.0, "kind": "intention", "data": {"hand": 0}}]
    u = P.urge_check(urges, reaches, paying=0, baseline_s=40.0)
    assert u["n"] == 1 and u["came_true"] == 1.0 and u["previous_reach"] == 0.0
    assert u["first_urge_s"] == 47.0 and u["first_habit_s"] == 60.0
    inner = [{"t": 1, "kind": "inner", "data": {"heard": "ba ba", "syllables": ["bo", "mi"]}},
             {"t": 2, "kind": "inner", "data": {"heard": "mi mi", "syllables": ["ba"]}},
             {"t": 3, "kind": "inner", "data": {"heard": "do do", "syllables": ["du"]}}]
    e = P.echo_check(inner)
    assert e["n"] == 3 and abs(e["first_matches"] - 2 / 3) < 1e-9


def test_tiny_runs_summarise_and_serialise(tmp_path):
    raw = [("hands", 1, True, P.run_hands(25, seed=1, baseline_s=10)),
           ("hands", 1, False, P.run_hands(25, seed=1, plastic=False, baseline_s=10)),
           ("voice", 1, True, P.run_voice(15, seed=1, baseline_s=5)),
           ("mind", 1, True, P.run_mind(15, seed=1, train_s=10))]
    res = P.summarize(raw)
    assert res["hands"]["plastic"][0]["paying"] in range(4)
    assert res["hands"]["plastic"][0]["paying"] == res["hands"]["frozen"][0]["paying"]
    assert res["mind"][0]["thought_rates"]["per_min"] > 0
    assert isinstance(P.report(res), str)
    json.loads(json.dumps(res, default=float))


def test_a_tiny_colony_experiment_runs_in_parts_and_is_collected(tmp_path):
    keep = tmp_path / "keep"
    kw = dict(days=2, seeds=(1,), day_length_s=12.0, population=2, cortex=100, keep=str(keep), verbose=False)
    a = C.main(policies=("evolving", "random"), **kw)
    b = C.main(policies=("static",), **kw)
    both = C.collect(keep, verbose=False)
    assert both["settings"]["policies"] == ["evolving", "random", "static"]
    assert [r["policy"] for r in both["runs"]] == ["evolving", "random", "static"]
    assert both["summary"]["static"] == b["summary"]["static"]
    assert both["summary"]["evolving"] == a["summary"]["evolving"]
    day = both["runs"][0]["daily"][0]
    assert not math.isnan(day["steer_food"]) and not math.isnan(day["steer_kin"])
    assert isinstance(C.report(both), str)
    rep = both["repeatability"]
    assert rep["n_pairs"] == 2 and len(rep["pairs"]) == 2 and rep["of_days"] == 1
    json.loads(json.dumps(both, default=float))
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    C.collect(keep, plot=str(tmp_path / "colony.png"), verbose=False)
    assert (tmp_path / "colony.png").stat().st_size > 10_000
