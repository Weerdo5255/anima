"""The Brain: a spiking network embedded in a body-regulating milieu.

The :class:`Brain` owns the spiking :class:`~anima.core.network.Network`
plus the slower, non-spiking systems that real brains also have: drives,
neuromodulators, the sleep/wake switch, the hippocampal store, the
cerebellar forward model and the striatal critic.  It is advanced in two
interleaved loops:

* ``step()`` — one millisecond of spiking dynamics (called 1000x per second);
* ``tick()`` — every 10 ms: update drives, neuromodulators and sleep, write
  their effects back into the network, schedule replay, store memories.

Nothing here waits for input.  If no one ever calls ``set_input`` the
network still fires (background noise + recurrent dynamics), the drives
still drift, the brain still gets bored, lonely, hungry and sleepy, and its
motor and vocal populations still win competitions and produce behaviour.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.network import Network
from ..core.plasticity import STDP, RewardSTDP
from .cerebellum import Cerebellum, CerebellumConfig
from .critic import Critic, CriticConfig
from .drives import DRIVE_NAMES, DriveConfig, Drives
from .memory import Hippocampus, HippocampusConfig
from .modulators import ModulatorConfig, Neuromodulators
from .sleep import SleepConfig, SleepController


@dataclass
class EffectConfig:
    """How strongly each neuromodulator acts on each kind of population."""

    sensory_gate: dict[str, float] = field(
        default_factory=lambda: {"awake": 1.0, "nrem": 0.12, "rem": 0.04})
    ach_sensory: float = 0.4  # sensory gain falls with acetylcholine
    cortex_adapt: float = 0.5  # extra fast adaptation when ACh is low
    slow_adapt: float = 2.0  # slow K+ adaptation per spike when ACh is low (up/down states)
    sleep_noise_drop: float = 0.2  # loss of tonic depolarising drive when ACh is low
    ne_gain: float = 0.35  # neural gain per unit noradrenaline (above 0.3)
    vigor: float = 3.0  # motor excitability per unit tonic dopamine (above 0.2)
    fatigue_inhibition: float = 4.0  # motor/vocal inhibition as sleep pressure builds
    # Paralysis in sleep is enforced downstream (the body's decoders ignore the
    # motor/vocal populations while asleep); the populations themselves are
    # only mildly inhibited so hippocampal replay can still reactivate them.
    atonia: float = 3.0
    curiosity_reward: float = 0.6  # reward per second for novelty when bored
    social_reward: float = 1.5  # reward per unit of loneliness relieved
    replay_interval_s: float = 0.6
    replay_duration_ms: float = 150.0
    replay_feedback: float = 8.0  # current injected into replayed action channels
    replay_da_gain: float = 0.6
    downscale_per_s: float = 0.0008  # NREM synaptic downscaling of cortical STDP synapses
    memory_reward_threshold: float = 0.05
    memory_surprise_threshold: float = 0.6


@dataclass
class InputPort:
    name: str
    role: str
    size: int
    channels: int
    max_current: float
    gated: bool
    labels: list[str]
    slice: slice
    neuron_gain: np.ndarray
    intensity: np.ndarray  # per neuron, in [0, 1]
    channel_values: np.ndarray | None = None  # per channel (cached mean)


@dataclass
class OutputGroup:
    name: str
    labels: list[str]
    channels: int


class Brain:
    def __init__(self, net: Network, spec: dict[str, Any] | None = None) -> None:
        if not net.built:
            net.build()
        self.net = net
        self.spec = spec or {}
        mods = self.spec.get("brain", {})
        self.effects = EffectConfig(**mods.get("effects", {}))
        self.drives = Drives(DriveConfig(**mods.get("drives", {})))
        self.sleep = SleepController(SleepConfig(**mods.get("sleep", {})))
        self.mods = Neuromodulators(ModulatorConfig(**mods.get("modulators", {})))
        self.hippocampus = Hippocampus(HippocampusConfig(**mods.get("hippocampus", {})))
        self.rng = np.random.default_rng(net.rng.integers(1 << 31))
        self.roles: dict[str, list[str]] = {}
        self.inputs: dict[str, InputPort] = {}
        self.outputs: dict[str, OutputGroup] = {}
        for pop in net.populations.values():
            role = pop.tags.get("role", "other")
            self.roles.setdefault(role, []).append(pop.name)
            if "input" in pop.tags:
                cfg = pop.tags["input"]
                labels = list(cfg.get("labels", [str(k) for k in range(pop.channels)]))
                self.inputs[pop.name] = InputPort(
                    name=pop.name, role=role, size=pop.size, channels=pop.channels,
                    max_current=float(cfg.get("max_current", 10.0)),
                    gated=bool(cfg.get("gated", role != "intero")), labels=labels,
                    slice=pop.slice, neuron_gain=0.7 + 0.6 * self.rng.random(pop.size),
                    intensity=np.zeros(pop.size))
            if "output" in pop.tags:
                cfg = pop.tags["output"]
                labels = list(cfg.get("labels", [str(k) for k in range(pop.channels)]))
                self.outputs[pop.name] = OutputGroup(pop.name, labels, pop.channels)
        self.intero_port = next((p for p in self.inputs.values() if p.role == "intero"), None)
        # cerebellum: predicts exteroceptive inputs from inputs + bodily state + actions
        self._extero = [p for p in self.inputs.values() if p.role != "intero"]
        n_out = sum(p.channels for p in self._extero)
        n_in = sum(p.channels for p in self.inputs.values()) + sum(
            o.channels for o in self.outputs.values())
        self.cerebellum = Cerebellum(n_in, n_out, CerebellumConfig(**mods.get("cerebellum", {})))
        n_feat = sum(p.channels for p in self.inputs.values())
        self.critic = Critic(n_feat, 0.01, CriticConfig(**mods.get("critic", {})))
        self._feedback = np.zeros(net.N)
        self._input_current = np.zeros(net.N)
        self._reward_acc = 0.0
        self.reward_log: deque[tuple[float, float, str]] = deque(maxlen=500)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.current_actions: dict[str, int] = {k: -1 for k in self.outputs}
        self._decisions: dict[str, tuple[float, dict[str, np.ndarray], int]] = {}
        self._replay_until = -1.0
        self._next_replay = 0.0
        self._replay_episode = None
        self._last_memory_t = -1.0
        self.replay_enabled = True
        self.attention = 0.0
        self.time_s = 0.0
        self.replays = 0
        self._pop_slices = {p.name: p.slice for p in net.populations.values()}
        self._base_noise = net.noise_sigma.copy()
        self._downscale_projs = [p for p in net.projections.values()
                                 if isinstance(p.rule, STDP)]
        self._reward_projs = [p for p in net.projections.values()
                              if isinstance(p.rule, RewardSTDP)]

    # ============================================================ I/O API
    def set_input(self, name: str, values: float | np.ndarray) -> None:
        """Drive an input population.  ``values`` may be a scalar, one value
        per channel, or one value per neuron; all in [0, 1]."""
        port = self.inputs[name]
        v = np.asarray(values, dtype=float)
        port.intensity[...] = self._expand(port.size, port.channels, v)
        if v.ndim == 0:
            port.channel_values = np.full(port.channels, float(v))
        elif v.size == port.channels:
            port.channel_values = v.astype(float).copy()
        else:
            port.channel_values = port.intensity.reshape(port.channels, -1).mean(axis=1)

    def get_input(self, name: str) -> np.ndarray:
        """Current per-channel intensities of an input port."""
        port = self.inputs[name]
        if port.channel_values is None:
            port.channel_values = port.intensity.reshape(port.channels, -1).mean(axis=1)
        return port.channel_values

    def set_feedback(self, name: str, values: float | np.ndarray) -> None:
        """Inject a current into any population (used for motor commitment)."""
        pop = self.net.populations[name]
        self._feedback[pop.slice] = self._expand(pop.size, pop.channels, values)

    @staticmethod
    def _expand(size: int, channels: int, values: float | np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if v.ndim == 0:
            return np.full(size, float(v))
        if v.size == channels:
            return np.repeat(v, size // channels)
        if v.size == size:
            return v
        raise ValueError(f"expected scalar, {channels} channel values or {size} neuron values")

    def channel_activity(self, name: str) -> np.ndarray:
        """Recent activity (~spikes per 30 ms per neuron) of each channel."""
        return self.net.channel_activity(name)

    def deliver_reward(self, amount: float, source: str = "") -> None:
        """Primary reward (+) or punishment (-) from the body or world."""
        if amount == 0:
            return
        self._reward_acc += amount
        self.reward_log.append((self.time_s, float(amount), source))

    def social_contact(self, strength: float = 1.0, valence: float = 0.0) -> float:
        """Someone engaged with us.  Relieves loneliness (rewarding in itself);
        ``valence`` adds approval (+) or disapproval (-)."""
        relief = self.drives.social_contact(strength)
        self.deliver_reward(self.effects.social_reward * relief + valence, "social")
        self.attention = max(self.attention, min(1.0, 0.5 + 0.5 * strength))
        if not self.sleep.awake:
            self.sleep.stimulate(0.5 * strength)
        return relief

    def note_action(self, group: str, channel: int) -> None:
        """The body committed to an action (efference copy)."""
        self.current_actions[group] = channel
        if channel >= 0:
            snap = {n: self.get_input(n).copy() for n in self.inputs}
            self._decisions[group] = (self.time_s, snap, channel)

    def end_action(self, group: str) -> None:
        self.current_actions[group] = -1

    # ============================================================ dynamics
    @property
    def stage(self) -> str:
        return self.sleep.stage

    @property
    def awake(self) -> bool:
        return self.sleep.awake

    def step(self) -> np.ndarray:
        return self.net.step()

    def tick(self, dt_ms: float = 10.0, energy: float | None = None) -> None:
        """Slow update; call every ``dt_ms`` of simulated time."""
        dt_s = dt_ms / 1000.0
        self.time_s += dt_s
        eff = self.effects
        net = self.net
        # ---- cerebellar prediction & surprise
        surprise = self._update_cerebellum()
        novelty = surprise
        # ---- drives
        cortex_rate = self._mean_rate("cortex")
        self.drives.update(dt_s, self.awake, self.drives.energy if energy is None else energy,
                           novelty, cortex_rate)
        if self.awake:
            self._reward_acc += eff.curiosity_reward * self.drives.boredom * novelty * dt_s
        # ---- sleep/wake
        before = self.sleep.stage
        changed = self.sleep.update(dt_s, self.time_s, self.drives.sleep_pressure,
                                    self.drives.circadian(), self.drives.cfg.day_length_s)
        if changed:
            self.events.append(("stage", {"stage": changed, "from": before, "t": self.time_s}))
        # ---- critic (ventral striatum) and neuromodulators
        reward = self._reward_acc
        self._reward_acc = 0.0
        rpe = None
        if self.critic.cfg.enabled:
            if abs(self.critic.tick_s - dt_s) > 1e-9:
                self.critic.set_tick(dt_s)
            feats = np.concatenate([self.get_input(n) for n in self.inputs]) if self.inputs \
                else np.zeros(0)
            rpe = self.critic.update(feats, reward, learn=self.awake and self.net.plasticity_enabled)
        self.mods.update(dt_s, reward, surprise, self.stage, self.attention, rpe=rpe)
        self.attention *= np.exp(-dt_s / 1.0)
        da = self.mods.dopamine
        # ---- memory: store salient waking moments
        if self.awake:
            self._maybe_remember(reward, surprise)
            self.hippocampus.decay(dt_s)
        # ---- replay and downscaling in slow-wave sleep
        replay_da = self._replay(dt_s)
        net.modulators["dopamine"] = float(np.clip(da + replay_da, -1.0, 1.0))
        net.modulators["acetylcholine"] = self.mods.acetylcholine
        if self.stage == "nrem" and eff.downscale_per_s > 0:
            f = np.float32(1.0 - eff.downscale_per_s * dt_s)
            for proj in self._downscale_projs:
                net.scale_weights(proj.name, f)
        # ---- write neuromodulatory effects into the network
        self._apply_effects()

    def _mean_rate(self, role: str) -> float:
        names = [n for n in self.roles.get(role, []) if not self.net.populations[n].inhibitory]
        if not names:
            return 0.0
        return float(np.mean([self.net.population_rate(n) for n in names]))

    def _update_cerebellum(self) -> float:
        x = [self.get_input(n) for n in self.inputs]
        for g, out in self.outputs.items():
            onehot = np.zeros(out.channels)
            ch = self.current_actions.get(g, -1)
            if ch >= 0:
                onehot[ch] = 1.0
            x.append(onehot)
        y = np.concatenate([self.get_input(p.name) for p in self._extero]) if self._extero \
            else np.zeros(0)
        if y.size == 0:
            return 0.0
        return self.cerebellum.step(np.concatenate(x), y)

    def _maybe_remember(self, reward: float, surprise: float) -> None:
        eff = self.effects
        if self.time_s - self._last_memory_t < 0.1:
            return
        if abs(reward) < eff.memory_reward_threshold and surprise < eff.memory_surprise_threshold:
            return
        recent = {g: d for g, d in self._decisions.items() if self.time_s - d[0] < 1.5}
        if recent:
            t0, snap, _ = max(recent.values(), key=lambda d: d[0])
            actions = {g: d[2] for g, d in recent.items()}
        else:
            snap = {n: self.get_input(n).copy() for n in self.inputs}
            actions = {}
        if self.hippocampus.encode(self.time_s, snap, actions, reward,
                                   surprise if surprise > eff.memory_surprise_threshold else 0.0):
            self._last_memory_t = self.time_s
            self.events.append(("memory", {"t": self.time_s, "reward": float(reward),
                                           "surprise": float(surprise), "actions": dict(actions)}))

    def _replay(self, dt_s: float) -> float:
        """Run hippocampal replay during NREM; returns extra dopamine."""
        eff = self.effects
        if self._replay_episode is not None and self.time_s >= self._replay_until:
            for g in self._replay_episode.actions:
                if g in self.outputs:
                    self.set_feedback(g, 0.0)
            self._replay_episode = None
        if self.stage != "nrem" or not self.replay_enabled:
            return 0.0
        if self._replay_episode is None and self.time_s >= self._next_replay:
            ep = self.hippocampus.sample(self.rng)
            self._next_replay = self.time_s + eff.replay_interval_s * (0.5 + self.rng.random())
            if ep is not None:
                self._replay_episode = ep
                self._replay_until = self.time_s + eff.replay_duration_ms / 1000.0
                self.replays += 1
                for g, ch in ep.actions.items():
                    if g in self.outputs:
                        fb = np.full(self.outputs[g].channels, -2.0)
                        fb[ch] = eff.replay_feedback
                        self.set_feedback(g, fb)
                self.events.append(("replay", {"t": self.time_s, "reward": ep.reward,
                                               "actions": dict(ep.actions), "episode_t": ep.t_s}))
        if self._replay_episode is not None:
            return float(np.clip(eff.replay_da_gain * self._replay_episode.reward, -0.5, 0.5))
        return 0.0

    def refresh(self) -> None:
        """Write current inputs and neuromodulatory effects into the network.

        Called automatically at the end of every :meth:`tick`; call it after
        ``set_input`` if new inputs must take effect before the next tick.
        """
        self._apply_effects()

    def _apply_effects(self) -> None:
        net, eff, m = self.net, self.effects, self.mods
        stage = self.stage
        gate = eff.sensory_gate.get(stage, 1.0) * ((1 - eff.ach_sensory) + eff.ach_sensory * m.acetylcholine)
        # ---- input currents (with thalamic gating), replay overrides
        I = self._input_current
        I[...] = 0.0
        if self.intero_port is not None:
            self.set_input(self.intero_port.name, self.drives.vector())
        ep = self._replay_episode
        for port in self.inputs.values():
            inten = port.intensity
            g = gate if port.gated else 1.0
            if ep is not None and port.name in ep.inputs and port.gated:
                inten = self._expand(port.size, port.channels, ep.inputs[port.name])
                g = 1.0
            I[port.slice] = inten * port.neuron_gain * (port.max_current * g)
        net.I_ext[...] = I + self._feedback
        # ---- gain / adaptation / excitability by role
        ach_def = max(0.0, 0.8 - m.acetylcholine) / 0.8
        ne_gain = 1.0 + eff.ne_gain * (m.noradrenaline - 0.3)
        for name in self.roles.get("cortex", []):
            s = self._pop_slices[name]
            net.adapt[s] = 1.0 + eff.cortex_adapt * ach_def
            net.gain[s] = ne_gain
            # Low acetylcholine (slow-wave sleep): less tonic drive and a slow
            # K+ current -> cortex alternates between silent "down" states and
            # bursts of activity ("up" states) at ~1 Hz.
            net.noise_sigma[s] = self._base_noise[s] * (1.0 - eff.sleep_noise_drop * ach_def)
            if not net.populations[name].inhibitory:
                net.slow_inc[s] = eff.slow_adapt * ach_def
        # fast homeostasis (intrinsic plasticity, synaptic scaling) would fight
        # the sleep state; pause it while asleep
        net.homeo_scale = 1.0 if self.awake else 0.0
        asleep = 0.0 if self.awake else 1.0
        fatigue = max(0.0, self.drives.fatigue - 0.5)
        motor_bias = (eff.vigor * (m.dopamine_tonic - 0.2) - eff.fatigue_inhibition * fatigue
                      - eff.atonia * asleep)
        for role in ("motor", "vocal"):
            for name in self.roles.get(role, []):
                if net.populations[name].inhibitory:
                    continue
                s = self._pop_slices[name]
                net.bias_mod[s] = motor_bias
                net.gain[s] = ne_gain
        net.params_changed()  # (a device engine uploads them before the next step)

    def action_threshold_scale(self) -> float:
        """Serotonin sets patience: low 5-HT releases actions more easily."""
        return 0.75 + 0.5 * self.mods.serotonin

    # ============================================================ views
    def region_rates(self) -> dict[str, float]:
        return {name: self.net.population_rate(name) for name in self.net.populations}

    def snapshot(self) -> dict[str, Any]:
        return {
            "t": self.time_s,
            "clock": self.drives.clock_string(),
            "stage": self.stage,
            "drives": self.drives.as_dict(),
            "modulators": self.mods.levels(),
            "rates": self.region_rates(),
            "memories": len(self.hippocampus),
            "replays": self.replays,
            "surprise": self.cerebellum.surprise,
            "value": self.critic.value,
        }

    def pop_events(self) -> list[tuple[str, dict[str, Any]]]:
        ev, self.events = self.events, []
        return ev

    # ============================================================ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "drives": self.drives.state_dict(),
            "sleep": self.sleep.state_dict(),
            "mods": self.mods.state_dict(),
            "hippocampus": self.hippocampus.state_dict(),
            "cerebellum": self.cerebellum.state_dict(),
            "critic": self.critic.state_dict(),
            "time_s": self.time_s,
            "replays": self.replays,
            "effects": asdict(self.effects),
            "rng": self.rng.bit_generator.state,
            # transient state, so a reloaded brain continues exactly
            "feedback": self._feedback.copy(),
            "intensity": {n: p.intensity.copy() for n, p in self.inputs.items()},
            "current_actions": dict(self.current_actions),
            "decisions": {g: [t, {k: v.copy() for k, v in snap.items()}, ch]
                          for g, (t, snap, ch) in self._decisions.items()},
            "attention": self.attention,
            "reward_acc": self._reward_acc,
            "next_replay": self._next_replay,
            "last_memory_t": self._last_memory_t,
            "net_modulators": dict(self.net.modulators),
        }

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.net.load_state_dict(s["net"])
        self.drives.load_state_dict(s["drives"])
        self.sleep.load_state_dict(s["sleep"])
        self.mods.load_state_dict(s["mods"])
        self.hippocampus.load_state_dict(s["hippocampus"])
        self.cerebellum.load_state_dict(s["cerebellum"])
        if "critic" in s:
            self.critic.load_state_dict(s["critic"])
        self.time_s = float(s["time_s"])
        self.replays = int(s["replays"])
        self.effects = EffectConfig(**s["effects"])
        self.rng.bit_generator.state = s["rng"]
        if "feedback" in s:
            self._feedback[...] = s["feedback"]
            for n, v in s["intensity"].items():
                if n in self.inputs:
                    self.inputs[n].intensity[...] = v
                    self.inputs[n].channel_values = None
            self.current_actions.update({k: int(v) for k, v in s["current_actions"].items()})
            self._decisions = {g: (float(d[0]), {k: np.asarray(v) for k, v in d[1].items()}, int(d[2]))
                               for g, d in s["decisions"].items()}
            self.attention = float(s["attention"])
            self._reward_acc = float(s["reward_acc"])
            self._next_replay = float(s["next_replay"])
            self._last_memory_t = float(s["last_memory_t"])
            self.net.modulators.update({k: float(v) for k, v in s["net_modulators"].items()})


__all__ = ["Brain", "EffectConfig", "InputPort", "OutputGroup", "DRIVE_NAMES"]
