"""Spiking core: neurons, network dynamics, learning rules."""

import numpy as np
import pytest

from anima.core import STDP, Hebbian, Network, RewardSTDP


def single(kind="RS", **kw):
    net = Network(seed=0)
    net.add_population(name="X", size=1, kind=kind, **kw)
    return net.build()


def test_regular_spiking_fi_curve_is_monotonic():
    rates = []
    for current in (2.0, 6.0, 12.0):
        net = single()
        net.I_ext[0] = current
        net.run(200)
        c0 = net.spike_count[0]
        net.run(1000)
        rates.append(net.spike_count[0] - c0)
    assert rates[0] == 0  # below rheobase
    assert 0 < rates[1] < rates[2]


def test_fast_spiking_fires_faster_than_regular_spiking():
    out = {}
    for kind in ("RS", "FS"):
        net = single(kind)
        net.I_ext[0] = 10.0
        net.run(1000)
        out[kind] = net.spike_count[0]
    assert out["FS"] > 2 * out["RS"]


def ei_network(seed=1, n=1000):
    net = Network(seed=seed)
    nE, nI = int(n * 0.8), n - int(n * 0.8)
    net.add_population(name="E", size=nE, kind="RS", noise=4.0, jitter=1.0)
    net.add_population(name="I", size=nI, kind="FS", noise=3.5, jitter=1.0)
    net.connect(pre="E", post="E", p=0.1, weight=0.015)
    net.connect(pre="E", post="I", p=0.2, weight=0.015)
    net.connect(pre="I", post="E", p=0.3, weight=0.08)
    net.connect(pre="I", post="I", p=0.2, weight=0.04)
    return net.build()


def test_spontaneous_activity_without_any_input():
    net = ei_network()
    net.run(1500)
    rE = net.population_rate("E")
    rI = net.population_rate("I")
    assert np.all(np.isfinite(net.v))
    assert 0.5 < rE < 30, rE  # alive but not epileptic
    assert rI > rE  # interneurons fire faster


def test_same_seed_same_spikes():
    a, b = ei_network(seed=3), ei_network(seed=3)
    a.run(300)
    b.run(300)
    np.testing.assert_array_equal(a.spike_count, b.spike_count)


def test_strong_inhibition_is_numerically_stable():
    """Exponential-Euler synapses must not overshoot E_GABA (old Euler bug)."""
    net = single()
    net.g_gaba[0] = 200.0  # absurdly strong inhibition
    net.run(50)
    assert net.spike_count[0] == 0
    assert net.v[0] > -80.0


def pair(rule, w=0.01):
    net = Network(seed=0)
    net.add_population(name="A", size=1)
    net.add_population(name="B", size=1)
    net.connect(pre="A", post="B", p=1.0, weight=w, weight_jitter=0.0, rule=rule, w_max=0.04)
    return net.build()


def pairings(net, dt_ms, n=15, dopamine=0.0):
    net.modulators["dopamine"] = dopamine
    a, b = net.pop("A").start, net.pop("B").start
    for _ in range(n):
        net.run(100)
        first, second = (a, b) if dt_ms > 0 else (b, a)
        net.v[first] = 40.0
        net.step()
        net.run(abs(dt_ms) - 1)
        net.v[second] = 40.0
        net.step()
    net.run(200)
    return float(net.weights("A->B")[0, 0])


def test_stdp_causal_potentiates_anticausal_depresses():
    assert pairings(pair(STDP()), +10) > 0.01
    assert pairings(pair(STDP()), -10) < 0.01


def test_reward_stdp_needs_dopamine():
    assert pairings(pair(RewardSTDP()), +10, dopamine=0.0) == pytest.approx(0.01)
    assert pairings(pair(RewardSTDP()), +10, dopamine=1.0) > 0.01
    assert pairings(pair(RewardSTDP()), +10, dopamine=-1.0) < 0.01


def test_reward_stdp_survives_paused_plasticity():
    """The lazily scaled eligibility must not overflow after a long pause."""
    net = pair(RewardSTDP(tau_e=300.0))
    net.plasticity_enabled = False
    net.run(60_000)
    net.plasticity_enabled = True
    pairings(net, +10, n=3, dopamine=1.0)
    assert np.isfinite(net.weights("A->B")).all()


def test_hebbian_is_order_symmetric():
    up = pairings(pair(Hebbian(norm_every=1e9)), +10)
    down = pairings(pair(Hebbian(norm_every=1e9)), -10)
    assert up > 0.01 and down > 0.01
    assert up == pytest.approx(down, rel=1e-6)


def test_fixed_indegree_and_channel_gain():
    net = Network(seed=0)
    net.add_population(name="P", size=40, channels=4)
    net.add_population(name="Q", size=30, channels=3)
    net.connect(pre="P", post="Q", p=0.5, weight=0.01, fixed_indegree=True,
                pre_channel_gain=[1, 0, 1, 1])
    net.build()
    m = net.mask("P->Q")
    assert (m.sum(axis=0) == 15).all()  # 0.5 * 30 eligible inputs, exactly
    assert not m[10:20].any()  # channel 1 silenced


def test_reward_stdp_normalisation_conserves_budget():
    net = Network(seed=0)
    net.add_population(name="P", size=20, noise=6.0)
    net.add_population(name="Q", size=10, noise=6.0)
    rule = RewardSTDP(a_minus=0.0, lr=0.5, normalize="post")
    net.connect(pre="P", post="Q", p=1.0, weight=0.01, rule=rule)
    net.build()
    before = net.weights("P->Q").sum(axis=0).copy()
    net.modulators["dopamine"] = 1.0
    net.run(500)
    after = net.weights("P->Q").sum(axis=0)
    np.testing.assert_allclose(after, before, rtol=1e-3)


def test_duplicate_projection_rejected():
    net = Network()
    net.add_population(name="A", size=2)
    net.connect(pre="A", post="A")
    with pytest.raises(ValueError):
        net.connect(pre="A", post="A", name="again")


def test_sparse_and_dense_stdp_agree():
    def build(sparse):
        net = Network(seed=11)
        net.add_population(name="E", size=200, kind="RS", noise=5.0, jitter=1.0)
        net.add_population(name="I", size=50, kind="FS", noise=3.5)
        net.connect(pre="E", post="E", p=0.1, weight=0.015,
                    rule=STDP(a_plus=0.01, a_minus=0.012, sparse=sparse))
        net.connect(pre="E", post="I", p=0.2, weight=0.015)
        net.connect(pre="I", post="E", p=0.3, weight=0.08)
        return net.build()

    a, b = build(True), build(False)
    a.run(1000)
    b.run(1000)
    assert a.projections["E->E"].rule._use_sparse and not b.projections["E->E"].rule._use_sparse
    np.testing.assert_allclose(a.weights("E->E"), b.weights("E->E"), rtol=1e-5, atol=1e-7)
    assert not np.allclose(a.weights("E->E"), build(True).weights("E->E"))  # it did learn
