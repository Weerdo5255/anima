"""Brain systems: drives, sleep, neuromodulators, memory, forward model, builder."""

import numpy as np

from anima.brain import (BrainBuilder, Drives, Hippocampus, Neuromodulators, SleepController,
                         build_brain, dumps_toml, infant_spec, load_spec, minimal_spec, save_spec)
from anima.brain.cerebellum import Cerebellum
from anima.brain.critic import Critic
from anima.brain.drives import DriveConfig


def test_drives_build_up_on_their_own():
    d = Drives()
    start = d.vector().copy()
    for _ in range(600):
        d.update(0.1, awake=True, energy=0.5, novelty=0.0, cortical_rate_hz=4.0)
    assert d.sleep_pressure > start[1]
    assert d.boredom > start[2]
    assert d.loneliness > start[3]
    relieved = d.social_contact(1.0)
    assert relieved > 0 and d.loneliness < 1.0


def test_two_process_model_sleeps_at_night():
    d = Drives(DriveConfig(day_length_s=1200))
    s = SleepController()
    wakes = []
    for _ in range(int(2 * 1200 / 0.5)):
        d.update(0.5, s.awake, 0.7, 0.0, 4.0)
        if s.update(0.5, d.t_s, d.sleep_pressure, d.circadian(), 1200) == "awake":
            wakes.append(d.phase)
    assert wakes, "never woke up"
    hour_wake = wakes[0] * 24
    assert 5.0 < hour_wake < 9.0, hour_wake


def test_loud_input_wakes_a_sleeper():
    s = SleepController()
    s.force("nrem", 0.0)
    s.time_asleep = 100.0
    s.stimulate(5.0)
    assert s.update(0.01, 1.0, 0.9, 0.0, 1200) == "awake"


def test_dopamine_tracks_prediction_error():
    m = Neuromodulators()
    m.update(0.01, reward=0.5, surprise=0.0, stage="awake")
    assert m.dopamine > 0.5
    for _ in range(300):
        m.update(0.01, reward=0.0, surprise=0.0, stage="awake")
    assert abs(m.dopamine) < 0.05
    m.update(0.01, reward=-0.5, surprise=0.0, stage="awake")
    assert m.dopamine < -0.5


def test_acetylcholine_falls_in_slow_wave_sleep():
    m = Neuromodulators()
    for _ in range(2000):
        m.update(0.01, 0.0, 0.0, "nrem")
    assert m.acetylcholine < 0.2
    for _ in range(2000):
        m.update(0.01, 0.0, 0.0, "rem")
    assert m.acetylcholine > 0.8


def test_hippocampus_prioritises_salient_episodes():
    h = Hippocampus()
    rng = np.random.default_rng(0)
    h.encode(0.0, {"x": np.ones(2)}, {"motor": 1}, reward=1.0, novelty=0.0)
    h.encode(1.0, {"x": np.zeros(2)}, {"motor": 0}, reward=0.1, novelty=0.0)
    picks = [h.sample(rng).reward for _ in range(10)]
    assert picks.count(1.0) >= 5


def test_cerebellum_learns_a_forward_model():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5)) * 0.3
    cb = Cerebellum(5, 3)
    errs = []
    xs = [rng.random(5) for _ in range(4000)]
    for k in range(len(xs)):
        y = A @ xs[k - cb.cfg.horizon_ticks] if k >= cb.cfg.horizon_ticks else np.zeros(3)
        cb.step(xs[k], y)
        errs.append(cb.error)
    assert errs[-1] < 0.3 * max(errs[50:200])


def test_critic_learns_to_predict_reward():
    c = Critic(2, tick_s=0.01)
    cue = np.array([1.0, 0.0])
    blank = np.zeros(2)
    for _trial in range(300):
        for _ in range(30):
            c.update(blank, 0.0)
        for k in range(50):  # cue, then reward 0.5 s later
            c.update(cue, 1.0 if k == 49 else 0.0)
    c.update(blank, 0.0)
    cue_delta = c.update(cue, 0.0)  # cue onset now produces a positive error
    assert cue_delta > 0.05


def test_builder_and_toml_round_trip(tmp_path):
    spec = infant_spec()
    path = tmp_path / "brain.toml"
    save_spec(spec, path)
    again = load_spec(path)
    assert [p["name"] for p in again["population"]] == [p["name"] for p in spec["population"]]
    assert len(again["projection"]) == len(spec["projection"])
    b1, b2 = build_brain(spec), build_brain(again)
    np.testing.assert_array_equal(b1.net.W, b2.net.W)
    assert "[[population]]" in dumps_toml(minimal_spec())


def test_fluent_builder():
    b = (BrainBuilder("tiny", seed=1)
         .input("intero", 20, channels=4, role="intero", gated=False)
         .population("cortex.E", 80, kind="RS", noise=4.0, role="cortex")
         .population("cortex.I", 20, kind="FS", noise=3.5, role="cortex")
         .output("motor", 30, channels=3, labels=["forward", "left", "right"], noise=3.5)
         .connect("cortex.E", "cortex.I", p=0.2, weight=0.015)
         .connect("cortex.I", "cortex.E", p=0.3, weight=0.08)
         .connect("cortex.E", "motor", p=0.2, weight=0.005, rule={"type": "reward_stdp"})
         .configure("drives", day_length_s=600.0))
    brain = b.build()
    assert brain.net.N == 150
    assert "motor" in brain.outputs and "intero" in brain.inputs
    for _ in range(50):
        for _ in range(10):
            brain.step()
        brain.tick(10.0)
    assert brain.net.population_rate("cortex.E") > 0


def test_replay_in_slow_wave_sleep_consolidates_a_rewarded_episode():
    """A remembered 'food on the left -> turn left -> reward' episode, replayed
    in NREM with its dopamine, strengthens bearing(left) -> left synapses."""
    brain = build_brain(infant_spec(seed=5))
    net = brain.net
    bearing = np.zeros(8)
    bearing[6] = 1.0  # +90 deg = food on the left
    brain.hippocampus.encode(0.0, {"sense.bearing": bearing}, {"motor": 1}, reward=1.0, novelty=0.0)
    pre = net.populations["sense.bearing"].channel_slice(6)
    post_l = net.populations["motor"].channel_slice(1)
    post_r = net.populations["motor"].channel_slice(2)

    def gap():
        W = net.W
        return float(W[pre, post_l].sum() - W[pre, post_r].sum())

    before = gap()
    brain.sleep.force("nrem", 0.0)
    brain.sleep.cfg.min_sleep_s = 1e9
    for _ in range(1500):  # 15 s of slow-wave sleep
        for _ in range(10):
            brain.step()
        brain.tick(10.0, 0.6)
    assert brain.replays > 5
    assert gap() > before
