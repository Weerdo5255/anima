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
from .hedonic import Hedonic, HedonicConfig
from .memory import Episode, Hippocampus, HippocampusConfig
from .modulators import ModulatorConfig, Neuromodulators
from .sleep import SleepConfig, SleepController


_GATES = ("vocal.gate",)  # "go" signals: recalling them would make it act, so they are not recalled


def _act_kind(group: str, role: str | None) -> str:
    if role == "vocal" or group.startswith("vocal"):
        return "speech"
    return "hand" if group == "hands" else "move"


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
    # plasticity of the hearing->speaking mirror while it speaks: x (1 + this x phasic dopamine), from 0 up
    hebbian_dopamine: float = 10.0


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
        self.hedonic = Hedonic(HedonicConfig(**mods.get("hedonic", {})))
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
        # pain and pleasure: what it did lately (to recall when it feels one)
        self.acts: deque[dict[str, Any]] = deque(maxlen=256)  # about a minute of moves and syllables
        self._act_id = 0
        self._recall_queue: deque[Episode] = deque()
        self._recalling = False
        self._recall_wait_until = -1.0
        self._frozen_until = -1.0
        self.recalls = 0
        self._pop_slices = {p.name: p.slice for p in net.populations.values()}
        self._base_noise = net.noise_sigma.copy()
        self._downscale_projs = [p for p in net.projections.values()
                                 if isinstance(p.rule, STDP)]
        self._reward_projs = [p for p in net.projections.values()
                              if isinstance(p.rule, RewardSTDP)]
        self._vocal_groups = {g for g in self.outputs if net.populations[g].tags.get("role") == "vocal"}

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

    def note_action(self, group: str, channel: int) -> int:
        """The body committed to an action (efference copy).  Returns the id
        of the act it is part of (a syllable notes its gate, consonant and
        vowel in the same tick: one act), which :meth:`feel` accepts as
        ``credit``."""
        self.current_actions[group] = channel
        snap = None
        if channel >= 0:
            snap = {n: self.get_input(n).copy() for n in self.inputs}
            self._decisions[group] = (self.time_s, snap, channel)
        kind = _act_kind(group, self.net.populations[group].tags.get("role") if group in self.net.populations else "")
        last = self.acts[-1] if self.acts else None
        if last is not None and last["kind"] == kind and abs(last["t"] - self.time_s) < 1e-9:
            last["actions"][group] = int(channel)  # another part of the same act
            return int(last["id"])
        self._act_id += 1
        self.acts.append({"id": self._act_id, "t": self.time_s, "kind": kind, "actions": {group: int(channel)},
                          "inputs": snap if snap is not None else {n: self.get_input(n).copy() for n in self.inputs}})
        return self._act_id

    def end_action(self, group: str) -> None:
        self.current_actions[group] = -1

    # ============================================================ pain and pleasure
    @property
    def frozen(self) -> bool:
        """For a moment it starts no new movement or sound: flinching from
        pain, or pausing to recall what it just did."""
        return self.time_s < self._frozen_until or self._recalling or bool(self._recall_queue)

    def feel(self, kind: str, intensity: float = 0.6, source: str = "", credit: Any = "auto") -> dict[str, Any]:
        """Feel pleasure (``kind="pleasure"``) or pain (``"pain"``) now; see
        :mod:`anima.brain.hedonic`.

        intensity: 0-1 (strength and duration of the teaching signal).
        credit: what it is for — the act(s) it recalls while the dopamine is
            there: ``"auto"`` (its last words if it spoke in the last few
            seconds, else its last movement), ``"speech"``, ``"movement"``,
            ``"now"`` (nothing recalled: only what it did in the last second
            or so is credited, through its eligibility traces), or a list of
            act ids (from :meth:`note_action`, e.g. the ``act`` of
            ``syllable`` events).
        Returns what was credited.  Call on the creature's own thread
        (:meth:`Agent.pleasure` / :meth:`Agent.pain` do).
        """
        h = self.hedonic
        cfg = h.cfg
        x = float(np.clip(intensity, 0.0, cfg.max_intensity))
        if kind not in ("pleasure", "pain"):
            raise ValueError(f"kind must be 'pleasure' or 'pain', not {kind!r}")
        acts = self._credit_acts(credit) if (cfg.recall and self.awake) else []
        duration = h.duration(x)
        if acts:
            duration = max(duration, len(acts) * cfg.recall_ms / 1000.0 + 0.25)
            for act in reversed(acts):  # the most recent first, as in reverse replay
                moves = {g: ch for g, ch in act["actions"].items() if g in self.outputs and g not in _GATES}
                self._recall_queue.append(Episode(act["t"], act["inputs"], moves,
                                                  x if kind == "pleasure" else -x, 0.0))
            self._recall_wait_until = self.time_s + cfg.recall_wait_s
            first = acts[0]
            self.hippocampus.encode(first["t"], first["inputs"], dict(first["actions"]),
                                    x if kind == "pleasure" else -x * cfg.pain_gain, 0.0)
            # (this is the memory of the moment; no second one of whatever it is doing now)
            self._last_memory_t = max(self._last_memory_t, self.time_s + duration)
        h.add(kind, x, duration, self.time_s, source)
        if kind == "pain":
            self._frozen_until = max(self._frozen_until, self.time_s + cfg.flinch_s * min(1.0, 0.4 + x))
            self.mods.arouse(cfg.arousal * x)
            self.attention = max(self.attention, x)
        if not self.awake:
            self.sleep.stimulate(cfg.wake * x * (1.0 if kind == "pain" else 1.0 / 3.0))
        what = [self.describe_act(a) for a in acts]
        label = (credit if isinstance(credit, str) else "acts") if acts else "now"
        info = {"intensity": round(x, 3), "source": source, "credit": label, "recalled": what,
                "seconds_ago": round(self.time_s - acts[-1]["t"], 2) if acts else None, "t": self.time_s}
        self.events.append((kind, info))
        return {"kind": kind, **info}

    def describe_act(self, act: dict[str, Any]) -> str:
        """A short name for an act: the syllable said, the movement made."""
        acts = act["actions"]
        if act["kind"] == "speech":
            c, v = acts.get("vocal.C", -1), acts.get("vocal.V", -1)
            cl = self.outputs["vocal.C"].labels if "vocal.C" in self.outputs else []
            vl = self.outputs["vocal.V"].labels if "vocal.V" in self.outputs else []
            return (cl[c] if 0 <= c < len(cl) else "") + (vl[v] if 0 <= v < len(vl) else "")
        g, ch = next(((g, ch) for g, ch in acts.items() if ch >= 0), (None, -1))
        if g is None:
            return act["kind"]
        labels = self.outputs[g].labels if g in self.outputs else []
        return labels[ch] if 0 <= ch < len(labels) else f"{g}:{ch}"

    def _credit_acts(self, credit: Any) -> list[dict[str, Any]]:
        """The acts (oldest first) a feeling is for."""
        if credit is None or credit == "now":
            return []
        if isinstance(credit, str):
            if credit == "speech":
                return self._last_words()
            if credit == "movement":
                return self._last_move()
            if credit == "auto":
                return self._last_words() or self._last_move()
            raise ValueError(f"credit must be one of auto, speech, movement, now or a list of act ids, not {credit!r}")
        ids = {int(i) for i in credit}
        return [a for a in self.acts if a["id"] in ids][-self.hedonic.cfg.recall_max:]

    def _last_words(self) -> list[dict[str, Any]]:
        cfg = self.hedonic.cfg
        said = [a for a in self.acts if a["kind"] == "speech"]
        if not said or self.time_s - said[-1]["t"] > cfg.recall_window_s:
            return []
        out = [said[-1]]
        for a in reversed(said[:-1]):
            if out[0]["t"] - a["t"] > cfg.recall_gap_s or len(out) >= cfg.recall_max:
                break
            out.insert(0, a)
        return out

    def _last_move(self) -> list[dict[str, Any]]:
        moved = [a for a in self.acts if a["kind"] in ("move", "hand")]
        if not moved or self.time_s - moved[-1]["t"] > self.hedonic.cfg.recall_move_s:
            return []
        return [moved[-1]]

    def _recall(self) -> None:
        """Replay the queued acts one after another (awake), each as a short
        burst: its senses as they were, and its articulators or muscles
        driven toward what it did, with the body paused."""
        if self._recalling and self._replay_episode is None:
            self._recalling = False  # the last recalled act just ended (see _replay)
        if self._recalling or not self._recall_queue:
            return
        if not self.awake:
            self._recall_queue.clear()
            return
        busy = any(ch >= 0 for ch in self.current_actions.values())
        if busy and self.time_s < self._recall_wait_until:
            return  # let the movement or syllable in progress finish first
        ep = self._recall_queue.popleft()
        self._replay_episode = ep
        self._replay_until = self.time_s + self.hedonic.cfg.recall_ms / 1000.0
        for g, ch in ep.actions.items():
            fb = np.full(self.outputs[g].channels, -2.0)
            if ch >= 0:
                fb[ch] = self.effects.replay_feedback
            self.set_feedback(g, fb)
        self._recalling = True
        self.recalls += 1

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
        felt = self.hedonic.step(dt_s)  # pain and pleasure
        if felt:
            self._reward_acc += felt
            self.reward_log.append((self.time_s, felt, "feel:pleasure" if felt > 0 else "feel:pain"))
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
        self._recall()
        net.modulators["dopamine"] = float(np.clip(da + replay_da, -1.0, 1.0))
        net.modulators["mirror"] = self._mirror_gate()
        net.modulators["acetylcholine"] = self.mods.acetylcholine
        if self.stage == "nrem" and eff.downscale_per_s > 0:
            f = np.float32(1.0 - eff.downscale_per_s * dt_s)
            for proj in self._downscale_projs:
                net.scale_weights(proj.name, f)
        # ---- write neuromodulatory effects into the network
        self._apply_effects()

    def _mirror_gate(self) -> float:
        """How fast the hearing->speaking mirror learns now (synapses with
        ``gate: "mirror"``).

        It learns *while it speaks* — what it hears then is mostly its own
        voice, the consequence of the articulation it is making — and while
        it relives speaking (a recalled or dreamt moment), but not while it
        only listens: otherwise every sound it hears would become linked to
        whatever it happens to say next, and a teacher's voice would blur the
        map rather than sharpen it.  Dopamine sets the rate: a burst
        multiplies it, a dip stops it (dopamine gates cortical LTP; pairing a
        sound with dopamine expands its cortical map: Bao, Chan & Merzenich
        2001).  So when it recalls a reply it is rewarded for, the sound it
        heard and the sound it made are bound much more strongly than in
        ordinary babbling, and replaying a moment that hurt binds nothing.
        """
        ep = self._replay_episode
        if ep is not None:
            if ep.reward < 0:
                return 0.0
            speaking = any(g in self._vocal_groups for g in ep.actions)
        else:
            speaking = any(self.current_actions.get(g, -1) >= 0 for g in self._vocal_groups)
        if not speaking:
            return 0.0
        k = self.effects.hebbian_dopamine
        return float(np.clip(1.0 + k * self.net.modulators.get("dopamine", 0.0), 0.0, 1.0 + k))

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
            "feelings": self.hedonic.levels(),
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
            # pain and pleasure (v0.6)
            "hedonic": self.hedonic.state_dict(),
            "acts": [_act_to_json(a) for a in self.acts],
            "act_id": self._act_id,
            "recall": {"queue": [_episode_to_json(e) for e in self._recall_queue],
                       "active": _episode_to_json(self._replay_episode) if self._recalling else None,
                       "until": self._replay_until, "wait_until": self._recall_wait_until,
                       "frozen_until": self._frozen_until, "recalls": self.recalls},
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
        if "hedonic" in s:
            self.hedonic.load_state_dict(s["hedonic"])
            self.acts = deque((_act_from_json(a) for a in s.get("acts", [])), maxlen=self.acts.maxlen)
            self._act_id = int(s.get("act_id", 0))
            r = s.get("recall", {})
            self._recall_queue = deque(_episode_from_json(e) for e in r.get("queue", []))
            if r.get("active") is not None:
                self._replay_episode = _episode_from_json(r["active"])
                self._replay_until = float(r["until"])
                self._recalling = True
            self._recall_wait_until = float(r.get("wait_until", -1.0))
            self._frozen_until = float(r.get("frozen_until", -1.0))
            self.recalls = int(r.get("recalls", 0))


def _act_to_json(a: dict[str, Any]) -> dict[str, Any]:
    return {**a, "inputs": {k: np.asarray(v, dtype=float).tolist() for k, v in a["inputs"].items()}}


def _act_from_json(a: dict[str, Any]) -> dict[str, Any]:
    return {"id": int(a["id"]), "t": float(a["t"]), "kind": str(a["kind"]),
            "actions": {k: int(v) for k, v in a["actions"].items()},
            "inputs": {k: np.asarray(v, dtype=float) for k, v in a["inputs"].items()}}


def _episode_to_json(e: Episode | None) -> dict[str, Any] | None:
    if e is None:
        return None
    return {"t_s": e.t_s, "inputs": {k: np.asarray(v, dtype=float).tolist() for k, v in e.inputs.items()},
            "actions": dict(e.actions), "reward": float(e.reward)}


def _episode_from_json(e: dict[str, Any]) -> Episode:
    return Episode(float(e["t_s"]), {k: np.asarray(v, dtype=np.float32) for k, v in e["inputs"].items()},
                   {k: int(v) for k, v in e["actions"].items()}, float(e.get("reward", 0.0)), 0.0)


__all__ = ["Brain", "EffectConfig", "InputPort", "OutputGroup", "DRIVE_NAMES"]
