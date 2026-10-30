"""The device engine (anima.core.engine), tested on the CPU with NumPy
(``device="emulate"``): it must reproduce the reference engine exactly where
it does the same arithmetic, closely where it batches plasticity, and save
to the same files."""

import json

import numpy as np
import pytest

from anima import Agent, infant_spec
from anima.brain.builder import build_network
from anima.core import Network, Population, Projection
from anima.core.device import DeviceError, resolve_device
from anima.core.plasticity import STDP, Hebbian, RewardSTDP

EXACT = {"dtype": np.float64, "noise": "host", "propagation": "sparse"}


def _pair(cortex=200, seed=3, **opts):
    spec = infant_spec(cortex_size=cortex, seed=seed)
    ref = build_network(spec)
    dev = build_network(spec, device="emulate", engine_options={**EXACT, **opts})
    for net in (ref, dev):
        net.I_ext[:] = 2.0
        net.params_changed()
    return ref, dev


def test_devices_resolve_and_explain_what_is_missing(monkeypatch):
    assert resolve_device("cpu").is_reference and resolve_device(None).is_reference
    assert str(resolve_device("emulate")) == "emulate"
    monkeypatch.setenv("ANIMA_DEVICE", "emulate")
    assert str(resolve_device(None)) == "emulate"
    monkeypatch.delenv("ANIMA_DEVICE")
    with pytest.raises(DeviceError):
        resolve_device("tpu")
    import importlib.util
    if importlib.util.find_spec("torch") is None and importlib.util.find_spec("cupy") is None:
        assert resolve_device("auto").is_reference  # no GPU library: auto falls back to the CPU
        with pytest.raises(DeviceError, match="pip install"):
            resolve_device("cuda")


def test_same_spikes_as_the_reference_engine_without_plasticity():
    ref, dev = _pair()
    assert dev.on_device and not ref.on_device
    assert np.array_equal(ref.W, dev.W) and np.array_equal(ref.M, dev.M)  # same wiring
    for net in (ref, dev):
        net.plasticity_enabled = False
    for _ in range(2000):  # includes intrinsic plasticity and two rounds of synaptic scaling
        ref.step()
        dev.step()
    assert np.array_equal(ref.spike_count, dev.spike_count)
    assert np.array_equal(ref.v, dev.v) and np.array_equal(ref.rate, dev.rate)
    assert ref.spike_count.sum() > 1000


@pytest.mark.parametrize("propagation", ["sparse", "dense"])
def test_learning_matches_the_reference_engine(propagation):
    ref, dev = _pair(propagation=propagation)
    for k in range(150):  # before the two drift apart
        for net in (ref, dev):
            net.modulators["dopamine"] = 0.4 if (k // 50) % 2 == 0 else -0.3
        ref.step()
        dev.step()
    assert np.array_equal(ref.spike_count, dev.spike_count)
    for name, p in ref.projections.items():
        if p.rule is None:
            continue
        a, b = ref.weights(name), dev.weights(name)
        tol = 1e-3 if isinstance(p.rule, STDP) else 1e-5  # STDP: applied per 10 ms window
        assert np.max(np.abs(a - b)) <= tol * np.max(np.abs(a)), name
        if isinstance(p.rule, RewardSTDP):  # the eligibility traces are exact
            np.testing.assert_allclose(dev.eligibility(name), ref.eligibility(name), rtol=1e-5, atol=1e-7)
    assert not np.array_equal(ref.weights("cortex.E->motor"), build_network(infant_spec(cortex_size=200, seed=3))
                              .weights("cortex.E->motor"))  # it did learn


def _two(rule, size=4, device=None, **kw):
    net = Network(seed=1, device=device, engine_options=EXACT if device else None)
    net.add_population(Population("A", size, noise=0.0))
    net.add_population(Population("B", size, noise=0.0))
    net.connect(Projection("A", "B", p=1.0, weight=0.01, w_max=0.04, rule=rule, **kw))
    return net.build()


@pytest.mark.parametrize("make", [lambda: STDP(), lambda: STDP(soft_bounds=False), lambda: Hebbian(rate=0.01),
                                  lambda: RewardSTDP(tau_e=200.0, apply_every=20.0, normalize="both"),
                                  lambda: RewardSTDP(tau_e=200.0, apply_every=20.0, w_min=0.002)])
def test_each_rule_on_controlled_spikes(make):
    """Pre and post neurons made to fire at chosen times (one pairing per 10 ms
    window): the batched device rules give what the reference rules give."""
    nets = [_two(make()), _two(make(), device="emulate")]
    w0 = nets[0].weights("A->B").copy()
    pattern = {3: [0, 1], 7: [5, 6], 14: [1], 17: [4], 31: [2, 7], 36: [3], 52: [6], 55: [0, 2]}
    for t in range(80):
        for net in nets:
            net.modulators["dopamine"] = 0.5 if t < 40 else -0.5
            net.I_ext[:] = 0.0
            for i in pattern.get(t, []):
                net.I_ext[i] = 400.0  # one strong millisecond: that neuron fires now
            net.params_changed()
            net.step()
    ref, dev = nets
    assert np.array_equal(ref.spike_count, dev.spike_count) and ref.spike_count.sum() >= 12
    np.testing.assert_allclose(dev.weights("A->B"), ref.weights("A->B"), rtol=2e-5, atol=1e-9)
    assert np.max(np.abs(ref.weights("A->B") - w0)) > 1e-5  # the rule did something


def test_a_creature_lives_learns_and_thinks_on_a_device(tmp_path):
    from anima.mind import ThoughtStream
    a = Agent.create(seed=5, device="emulate", cortex_size=400)
    assert a.device == "emulate"
    mind = ThoughtStream(a, tmp_path / "t.jsonl")
    w0 = a.brain.net.weights("cortex.E->motor")
    a.run(seconds=3.0)
    assert a.body.stats["moves"] > 0 and a.latest["rates"]["cortex.E"] > 0
    assert not np.array_equal(w0, a.brain.net.weights("cortex.E->motor"))
    assert len(mind.recent(50)) > 0
    mind.close()
    c = a.clone()
    assert c.device == "emulate"
    with pytest.raises(ValueError):  # device state is read through copies; write with methods
        a.brain.net.v[0] = 1.0


def test_files_move_between_cpu_and_device(tmp_path):
    a = Agent.create(seed=4, cortex_size=300)
    a.run(seconds=1.0)
    a.save(tmp_path / "cpu.npz")
    b = Agent.load(tmp_path / "cpu.npz", device="emulate")  # (float32 on the device)
    for k in ("spike_count", "W"):
        np.testing.assert_array_equal(getattr(a.brain.net, k), getattr(b.brain.net, k))
    for k in ("v", "rate"):
        np.testing.assert_allclose(getattr(b.brain.net, k), getattr(a.brain.net, k), rtol=1e-6, atol=1e-6)
    b.run(seconds=1.0)
    b.save(tmp_path / "dev.npz")
    c = Agent.load(tmp_path / "dev.npz")  # back on the CPU
    sb, sc = b.brain.net.state_dict(), c.brain.net.state_dict()
    assert set(sb) == set(sc) and set(sb["rules"]) == set(sc["rules"])
    for k in ("v", "W", "spike_count", "bias_homeo"):
        np.testing.assert_allclose(sc[k], sb[k], rtol=1e-6)
    for name, st in sb["rules"].items():
        for key, val in st.items():
            np.testing.assert_allclose(np.asarray(sc["rules"][name][key], float), np.asarray(val, float), rtol=1e-6)
    c.run(seconds=0.5)
    assert c.t_ms == b.t_ms + 500.0


def test_weight_methods_work_on_both_engines():
    for device in (None, "emulate"):
        net = build_network(infant_spec(cortex_size=200, seed=2), device=device)
        name = "cortex.E->motor"
        w = net.weights(name).copy()
        net.scale_weights(name, 0.5)
        np.testing.assert_allclose(net.weights(name), w * np.float32(0.5), rtol=1e-6)
        net.set_weights(name, w)
        np.testing.assert_array_equal(net.weights(name), w)
        net.jitter_weights(0.1, np.random.default_rng(0))
        changed = net.weights(name)
        assert not np.array_equal(changed, w) and np.all(changed[~net.mask(name)] == 0)
        d = net.drive([net.projections[name]], "motor")
        assert d.shape == (net.populations["motor"].size,)
        net.reseed(11)


def test_a_colony_runs_on_a_device(tmp_path):
    from anima.evolve import Colony, ColonyConfig
    col = Colony.create(tmp_path / "col", ColonyConfig(population=2, cortex=150, day_length_s=40.0, thoughts=False),
                        printer=None, device="emulate")
    assert col.info(0)["device"] == "emulate"
    col.run(seconds=2.0)
    col.save()
    col.close()
    again = Colony.load(tmp_path / "col", printer=None)  # and on the CPU
    again.run(seconds=1.0)
    again.close()


def test_gpu_check_in_emulation():
    from anima.benchmark import gpu_check
    lines = []
    res = gpu_check("emulate", say=lines.append, quick=True)
    assert res["ok"] and res["agreement"]["identical"]
    assert any("identical" in s for s in lines)
    json.dumps(res, default=float)
