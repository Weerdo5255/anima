"""Event-driven spiking network simulator.

Design notes
------------
* All neurons of all populations live in flat arrays; a population is a
  contiguous index range.  A projection between two populations is therefore
  a rectangular *view* into one global weight matrix ``W[pre, post]``.  This
  keeps the per-millisecond work to a handful of NumPy calls, regardless of
  how many populations or projections a brain has.
* Synapses are conductance based (AMPA, NMDA with Mg2+ block, GABA-A).
  Conductance synapses saturate at their reversal potentials, which makes
  recurrent networks far more stable than current-based ones.
* Spike propagation is event driven: only the rows of ``W`` belonging to
  neurons that fired this step are summed.
* Dale's law: a neuron is either excitatory (all outgoing synapses AMPA/NMDA)
  or inhibitory (all outgoing synapses GABA-A).
* Time step is 1 ms by default.  ``step()`` advances one step and never
  waits for anything; the simulation is a free-running dynamical system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .neurons import V_PEAK, make_params

E_AMPA = 0.0
E_NMDA = 0.0
E_GABA = -75.0
TAU_AMPA = 5.0
TAU_NMDA = 100.0
TAU_GABA = 7.0
TAU_TRACE = 20.0  # STDP trace time constant (ms)
TAU_READOUT = 30.0  # activity trace used by motor decoders (ms)
TAU_RATE = 1000.0  # firing-rate estimate used by homeostasis (ms)
TAU_SLOW = 500.0  # slow, Ca2+-activated K+ adaptation current (ms)

INHIBITORY_KINDS = {"FS", "LTS"}


@dataclass
class Population:
    """A group of neurons sharing a cell type and role.

    Parameters
    ----------
    name: unique name, e.g. ``"cortex.E"``.
    size: number of neurons.
    kind: Izhikevich cell type (see :data:`anima.core.neurons.PRESETS`).
    inhibitory: override Dale sign (defaults to True for FS/LTS cells).
    noise: standard deviation of the per-millisecond background current.
        This is the source of *spontaneous* activity: ion-channel noise and
        the barrage of unmodelled synaptic input every real neuron receives.
    bias: constant input current (intrinsic excitability).
    jitter: cell-to-cell parameter heterogeneity in [0, 1].
    nmda_ratio: fraction of incoming excitatory conductance that is also
        delivered through slow NMDA receptors (persistent activity).
    target_rate: if set (Hz), intrinsic homeostatic plasticity slowly tunes
        each neuron's excitability toward this firing rate.
    channels: split the population into this many equal groups (e.g. one
        group per action or per phoneme).
    """

    name: str
    size: int
    kind: str = "RS"
    inhibitory: bool | None = None
    noise: float = 0.0
    bias: float = 0.0
    jitter: float = 0.0
    nmda_ratio: float = 0.0
    target_rate: float | None = None
    channels: int = 1
    tags: dict[str, Any] = field(default_factory=dict)
    start: int = -1
    stop: int = -1

    def __post_init__(self) -> None:
        if self.inhibitory is None:
            self.inhibitory = self.kind in INHIBITORY_KINDS
        if self.size <= 0:
            raise ValueError(f"population {self.name!r} must have size > 0")
        if self.size % self.channels:
            raise ValueError(
                f"population {self.name!r}: size {self.size} not divisible by channels {self.channels}"
            )

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def channel_size(self) -> int:
        return self.size // self.channels

    def channel_slice(self, k: int) -> slice:
        """Global index range of channel ``k``."""
        if not 0 <= k < self.channels:
            raise IndexError(f"{self.name}: channel {k} out of range")
        s = self.start + k * self.channel_size
        return slice(s, s + self.channel_size)


@dataclass
class Projection:
    """Synaptic connections from one population to another.

    topology:
        ``"random"``    each pair connected with probability ``p``;
        ``"one_to_one"`` neuron i -> neuron i (equal sizes);
        ``"channels"``  channel k -> channel k only, with probability ``p``
                        (both populations need the same channel count);
        ``"cross_channels"`` channel k -> every channel except k.

    pre_channel_gain / post_channel_gain:
        Optional per-channel multipliers of the initial weights.  A zero
        removes the connections of that channel entirely, e.g. let only the
        "loneliness" channel of interoception drive the vocal system.
    """

    pre: str
    post: str
    p: float = 0.1
    weight: float = 0.01
    weight_jitter: float = 0.5
    w_max: float | None = None
    topology: str = "random"
    rule: Any = None  # a plasticity rule instance (see anima.core.plasticity)
    scaling: bool = False  # homeostatic synaptic scaling of these synapses
    autapses: bool = False
    pre_channel_gain: list[float] | None = None
    post_channel_gain: list[float] | None = None
    fixed_indegree: bool = False  # exactly round(p * n) inputs per target neuron
    name: str = ""
    # filled at build time
    pre_pop: Population | None = None
    post_pop: Population | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.pre}->{self.post}"
        if self.w_max is None:
            self.w_max = 4.0 * self.weight if self.weight > 0 else 1.0

    @property
    def pre_slice(self) -> slice:
        assert self.pre_pop is not None
        return self.pre_pop.slice

    @property
    def post_slice(self) -> slice:
        assert self.post_pop is not None
        return self.post_pop.slice


class Network:
    """A free-running spiking neural network."""

    def __init__(self, seed: int | None = None, dt: float = 1.0) -> None:
        if dt != 1.0:
            # The Izhikevich half-step integration and all time constants are
            # tuned for 1 ms; keep the API honest instead of silently wrong.
            raise ValueError("anima currently supports dt = 1.0 ms only")
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.populations: dict[str, Population] = {}
        self.projections: dict[str, Projection] = {}
        self.built = False
        self.t = 0.0  # ms
        self.step_count = 0
        self.plasticity_enabled = True
        #: Scalar neuromodulator levels read by plasticity rules
        #: (written by the Brain).  ``dopamine`` is phasic (signed).
        self.modulators: dict[str, float] = {"dopamine": 0.0, "acetylcholine": 1.0}
        self.slow_every = 10  # ms between slow (plasticity/homeostasis) ticks
        self.homeostasis_enabled = True
        self.ip_rate = 0.02  # intrinsic plasticity rate (current units / s / Hz)
        self.ip_max = 6.0
        self.scaling_every = 1000  # ms
        self.scaling_rate = 0.05
        self._post_step_hooks: list[Callable[["Network", np.ndarray], None]] = []

    # ------------------------------------------------------------------ build
    def add_population(self, pop: Population | None = None, **kwargs: Any) -> Population:
        if self.built:
            raise RuntimeError("cannot add populations after build()")
        if pop is None:
            pop = Population(**kwargs)
        if pop.name in self.populations:
            raise ValueError(f"duplicate population {pop.name!r}")
        self.populations[pop.name] = pop
        return pop

    def connect(self, proj: Projection | None = None, **kwargs: Any) -> Projection:
        if self.built:
            raise RuntimeError("cannot add projections after build()")
        if proj is None:
            proj = Projection(**kwargs)
        for end in (proj.pre, proj.post):
            if end not in self.populations:
                raise KeyError(f"projection {proj.name!r}: unknown population {end!r}")
        if proj.name in self.projections:
            raise ValueError(f"duplicate projection {proj.name!r}")
        for other in self.projections.values():
            if other.pre == proj.pre and other.post == proj.post:
                raise ValueError(
                    f"{proj.pre}->{proj.post} already connected by {other.name!r}; "
                    "combine them into one projection"
                )
        self.projections[proj.name] = proj
        return proj

    def build(self) -> "Network":
        if self.built:
            return self
        rng = self.rng
        n = 0
        for pop in self.populations.values():
            pop.start, pop.stop = n, n + pop.size
            n += pop.size
        self.N = N = n
        self.v = np.full(N, -65.0)
        self.a = np.empty(N)
        self.b = np.empty(N)
        self.c = np.empty(N)
        self.d_base = np.empty(N)
        self.is_inh = np.zeros(N, dtype=bool)
        self.noise_sigma = np.zeros(N)
        self.bias_base = np.zeros(N)
        self.nmda_ratio = np.zeros(N)
        self.target_rate = np.full(N, np.nan)
        for pop in self.populations.values():
            s = pop.slice
            a, b, c, d = make_params(pop.kind, pop.size, rng, pop.jitter)
            self.a[s], self.b[s], self.c[s], self.d_base[s] = a, b, c, d
            self.is_inh[s] = bool(pop.inhibitory)
            self.noise_sigma[s] = pop.noise
            self.bias_base[s] = pop.bias
            self.nmda_ratio[s] = pop.nmda_ratio
            if pop.target_rate is not None:
                self.target_rate[s] = pop.target_rate
        self.u = self.b * self.v
        # neuromodulatory multipliers (set by the brain)
        self.gain = np.ones(N)  # multiplies synaptic current
        self.adapt = np.ones(N)  # multiplies spike-triggered adaptation d
        self.bias_mod = np.zeros(N)  # additive excitability shift
        self.bias_homeo = np.zeros(N)  # intrinsic plasticity
        self.I_ext = np.zeros(N)  # external drive written by the body
        # All exponentially decaying per-neuron state lives in one 2-D array
        # so it decays with a single vectorised multiply per step.
        self._S = np.zeros((7, N))
        self.g_ampa, self.g_gaba, self.g_nmda = self._S[0], self._S[1], self._S[2]
        self.trace, self.readout, self.rate = self._S[3], self._S[4], self._S[5]
        # slow adaptation current: each spike adds ``slow_inc``; it decays with
        # TAU_SLOW.  Zero in waking (acetylcholine blocks it); in slow-wave sleep
        # it makes cortex alternate between up and down states.
        self.w_slow = self._S[6]
        self.slow_inc = np.zeros(N)
        #: multiplier on both homeostatic mechanisms (intrinsic plasticity and
        #: synaptic scaling); the brain sets it to 0 while asleep, because these
        #: fast set-point controllers would otherwise "correct" the sleep state
        self.homeo_scale = 1.0
        self.has_nmda = bool(np.any(self.nmda_ratio > 0))
        self.spike_count = np.zeros(N, dtype=np.int64)
        self.fired = np.zeros(N, dtype=bool)
        self.last_spikes = np.zeros(0, dtype=np.int64)
        self.W = np.zeros((N, N), dtype=np.float32)
        self.M = np.zeros((N, N), dtype=bool)
        for proj in self.projections.values():
            proj.pre_pop = self.populations[proj.pre]
            proj.post_pop = self.populations[proj.post]
            gain = self._channel_gain(proj)
            mask = self._make_mask(proj, rng, None if gain is None else gain > 0)
            w = proj.weight * (1.0 + proj.weight_jitter * (2.0 * rng.random(mask.shape) - 1.0))
            if gain is not None:
                w = w * gain
            w = np.clip(w, 0.0, proj.w_max) * mask
            ps, qs = proj.pre_slice, proj.post_slice
            self.M[ps, qs] = mask
            self.W[ps, qs] = w.astype(np.float32)
        self._plastic = [p for p in self.projections.values() if p.rule is not None]
        for proj in self._plastic:
            proj.rule.setup(self, proj)
        self._pop_list = list(self.populations.values())
        self._edges = np.array([p.start for p in self._pop_list] + [N])
        taus = np.array([TAU_AMPA, TAU_GABA, TAU_NMDA, TAU_TRACE, TAU_READOUT, TAU_RATE,
                         TAU_SLOW])
        self._decay = np.exp(-self.dt / taus)[:, None]
        self._noise_buf = np.empty((0, N))
        self._noise_pos = 0
        self.built = True
        return self

    @staticmethod
    def _channel_gain(proj: Projection) -> np.ndarray | None:
        pre, post = proj.pre_pop, proj.post_pop
        assert pre is not None and post is not None
        if proj.pre_channel_gain is None and proj.post_channel_gain is None:
            return None
        g = np.ones((pre.size, post.size))
        if proj.pre_channel_gain is not None:
            pg = np.asarray(proj.pre_channel_gain, dtype=float)
            if pg.size != pre.channels:
                raise ValueError(f"{proj.name}: pre_channel_gain needs {pre.channels} values")
            g *= np.repeat(pg, pre.channel_size)[:, None]
        if proj.post_channel_gain is not None:
            qg = np.asarray(proj.post_channel_gain, dtype=float)
            if qg.size != post.channels:
                raise ValueError(f"{proj.name}: post_channel_gain needs {post.channels} values")
            g *= np.repeat(qg, post.channel_size)[None, :]
        return g

    def _make_mask(self, proj: Projection, rng: np.random.Generator,
                   eligible: np.ndarray | None = None) -> np.ndarray:
        pre, post = proj.pre_pop, proj.post_pop
        assert pre is not None and post is not None
        shape = (pre.size, post.size)
        topo = proj.topology
        if isinstance(topo, np.ndarray):
            if topo.shape != shape:
                raise ValueError(f"{proj.name}: mask shape {topo.shape} != {shape}")
            mask = topo.astype(bool)
            return mask if eligible is None else mask & eligible
        if topo == "one_to_one":
            if pre.size != post.size:
                raise ValueError(f"{proj.name}: one_to_one needs equal sizes")
            mask = np.eye(pre.size, dtype=bool)
            return mask if eligible is None else mask & eligible
        if topo == "random":
            allowed = np.ones(shape, dtype=bool)
        elif topo in ("channels", "cross_channels"):
            if pre.channels != post.channels:
                raise ValueError(f"{proj.name}: {topo} needs equal channel counts")
            k = pre.channels
            same = np.kron(
                np.eye(k, dtype=bool), np.ones((pre.channel_size, post.channel_size), dtype=bool)
            )
            allowed = same if topo == "channels" else ~same
        else:
            raise ValueError(f"{proj.name}: unknown topology {topo!r}")
        if eligible is not None:
            allowed &= eligible
        if proj.pre == proj.post and not proj.autapses:
            np.fill_diagonal(allowed, False)
        if not proj.fixed_indegree:
            return allowed & (rng.random(shape) < proj.p)
        # every postsynaptic neuron gets exactly round(p * candidates) inputs,
        # which removes the chance differences in total drive that would
        # otherwise make one channel of a competition win systematically
        mask = np.zeros(shape, dtype=bool)
        for j in range(post.size):
            cand = np.flatnonzero(allowed[:, j])
            k = int(round(proj.p * cand.size))
            if k:
                mask[rng.choice(cand, size=k, replace=False), j] = True
        return mask

    # ------------------------------------------------------------- dynamics
    def _noise(self) -> np.ndarray:
        if self._noise_pos >= len(self._noise_buf):
            self._noise_buf = self.rng.standard_normal((200, self.N))
            self._noise_pos = 0
        row = self._noise_buf[self._noise_pos]
        self._noise_pos += 1
        return row

    def step(self) -> np.ndarray:
        """Advance the network by one time step; return indices that spiked."""
        if not self.built:
            self.build()
        v, u = self.v, self.u
        # Synaptic conductances relax v toward their combined reversal
        # potential.  That linear part is integrated exactly (exponential
        # Euler), which is stable for arbitrarily large conductances; plain
        # Euler overshoots E_GABA under strong inhibition and makes neurons
        # fire every step.  Over a half step h:
        #     v <- v * exp(-hG) + (sum_k g_k E_k) * (1 - exp(-hG)) / G
        G = self.g_ampa + self.g_gaba
        GE = self.g_gaba * E_GABA  # (E_AMPA = E_NMDA = 0 mV)
        if self.has_nmda:
            mg = np.exp(-0.062 * v)
            mg *= 0.28
            mg += 1.0
            G += self.g_nmda / mg  # Mg2+ block
        G *= self.gain
        GE *= self.gain
        G += 1e-9
        G *= -0.5 * self.dt
        em = np.expm1(G)  # exp(-hG) - 1
        GE *= em
        GE /= G  # G holds -hG here
        GE *= 0.5 * self.dt  # -> sum_k g_k E_k * (1 - exp(-hG)) / G
        em += 1.0  # = exp(-hG)
        # half of the non-synaptic drive per 0.5 ms sub-step
        J = self.noise_sigma * self._noise()
        J += self.I_ext
        J += self.bias_base
        J += self.bias_mod
        J += self.bias_homeo
        J += 140.0
        J -= self.w_slow
        J -= u
        J *= 0.5
        t = np.empty_like(v)
        # Izhikevich: two 0.5 ms sub-steps (operator splitting)
        for _ in range(2):
            v *= em
            v += GE
            np.multiply(v, 0.02, out=t)  # 0.5 * (0.04 v^2 + 5 v) = v (0.02 v + 2.5)
            t += 2.5
            t *= v
            t += J
            v += t
            np.minimum(v, V_PEAK, out=v)
        np.maximum(v, -120.0, out=v)
        u += self.a * (self.b * v - u)
        fired = v >= V_PEAK
        idx = np.flatnonzero(fired)
        if idx.size:
            v[idx] = self.c[idx]
            u[idx] += self.d_base[idx] * self.adapt[idx]
        # conductances, traces, readout and rate all decay in one multiply
        self._S *= self._decay
        if idx.size:
            inh_mask = self.is_inh[idx]
            exc = idx[~inh_mask]
            inh = idx[inh_mask]
            if exc.size:
                inc = self.W[exc].sum(axis=0)
                self.g_ampa += inc
                if self.has_nmda:
                    self.g_nmda += inc * self.nmda_ratio
            if inh.size:
                self.g_gaba += self.W[inh].sum(axis=0)
            if self.plasticity_enabled and self._plastic:
                self._apply_spike_plasticity(idx)
            self.trace[idx] += 1.0
            self.readout[idx] += 1.0
            self.rate[idx] += 1000.0 / TAU_RATE
            self.w_slow[idx] += self.slow_inc[idx]
            self.spike_count[idx] += 1
        self.fired = fired
        self.last_spikes = idx
        self.t += self.dt
        self.step_count += 1
        if self.step_count % self.slow_every == 0:
            self._slow_tick()
        for hook in self._post_step_hooks:
            hook(self, idx)
        return idx

    def run(self, ms: float) -> None:
        for _ in range(int(round(ms / self.dt))):
            self.step()

    def _split_spikes(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        bounds = np.searchsorted(idx, self._edges)
        return {
            pop.name: idx[bounds[k] : bounds[k + 1]]
            for k, pop in enumerate(self._pop_list)
            if bounds[k + 1] > bounds[k]
        }

    def _apply_spike_plasticity(self, idx: np.ndarray) -> None:
        by_pop = self._split_spikes(idx)
        for proj in self._plastic:
            pre = by_pop.get(proj.pre)
            post = by_pop.get(proj.post)
            if pre is None and post is None:
                continue
            proj.rule.on_spikes(
                self,
                proj,
                None if pre is None else pre - proj.pre_pop.start,
                None if post is None else post - proj.post_pop.start,
            )

    def _slow_tick(self) -> None:
        dt_ms = self.slow_every * self.dt
        if self.plasticity_enabled:
            for proj in self._plastic:
                proj.rule.on_slow(self, proj, dt_ms)
        if self.homeostasis_enabled:
            has_target = ~np.isnan(self.target_rate)
            if np.any(has_target):
                err = np.where(has_target, self.target_rate - self.rate, 0.0)
                self.bias_homeo += (self.ip_rate * self.homeo_scale) * err * (dt_ms / 1000.0)
                np.clip(self.bias_homeo, -self.ip_max, self.ip_max, out=self.bias_homeo)
            if self.step_count % self.scaling_every == 0:
                self._synaptic_scaling()

    def _synaptic_scaling(self) -> None:
        """Multiplicative synaptic scaling (Turrigiano 1998) of flagged projections."""
        if self.homeo_scale <= 0:
            return
        for proj in self.projections.values():
            if not proj.scaling:
                continue
            qs = proj.post_slice
            target = self.target_rate[qs]
            if np.all(np.isnan(target)):
                continue
            rate = self.rate[qs]
            target = np.where(np.isnan(target), rate, target)
            factor = 1.0 + (self.scaling_rate * self.homeo_scale) * (target - rate) / np.maximum(target, 0.5)
            factor = np.clip(factor, 0.9, 1.1).astype(np.float32)
            blk = self.W[proj.pre_slice, qs]
            blk *= factor[None, :]
            np.minimum(blk, proj.w_max, out=blk)

    # -------------------------------------------------------------- helpers
    def pop(self, name: str) -> Population:
        return self.populations[name]

    def inject(self, name: str, current: float | np.ndarray, channel: int | None = None) -> None:
        """Set the external input current of a population (or one channel)."""
        pop = self.populations[name]
        s = pop.slice if channel is None else pop.channel_slice(channel)
        self.I_ext[s] = current

    def channel_activity(self, name: str, trace: str = "readout") -> np.ndarray:
        """Mean activity trace per channel of a population."""
        pop = self.populations[name]
        arr = getattr(self, trace)[pop.slice]
        return arr.reshape(pop.channels, pop.channel_size).mean(axis=1)

    def population_rate(self, name: str) -> float:
        return float(self.rate[self.populations[name].slice].mean())

    def weights(self, proj_name: str) -> np.ndarray:
        """View of a projection's weight block (pre x post)."""
        proj = self.projections[proj_name]
        return self.W[proj.pre_slice, proj.post_slice]

    def mask(self, proj_name: str) -> np.ndarray:
        proj = self.projections[proj_name]
        return self.M[proj.pre_slice, proj.post_slice]

    def add_post_step_hook(self, fn: Callable[["Network", np.ndarray], None]) -> None:
        self._post_step_hooks.append(fn)

    def summary(self) -> str:
        lines = [f"Network: {self.N if self.built else '?'} neurons, "
                 f"{len(self.populations)} populations, {len(self.projections)} projections"]
        for pop in self.populations.values():
            sign = "inh" if pop.inhibitory else "exc"
            lines.append(f"  {pop.name:<22} {pop.size:>5} {pop.kind:<4} {sign}"
                         + (f" x{pop.channels}ch" if pop.channels > 1 else ""))
        if self.built:
            for proj in self.projections.values():
                nsyn = int(self.M[proj.pre_slice, proj.post_slice].sum())
                rule = type(proj.rule).__name__ if proj.rule is not None else "static"
                lines.append(f"  {proj.name:<34} {nsyn:>7} syn  {rule}")
        return "\n".join(lines)

    # ---------------------------------------------------------- persistence
    _STATE_ARRAYS = (
        "v", "u", "g_ampa", "g_nmda", "g_gaba", "trace", "readout", "rate",
        "bias_homeo", "W", "spike_count", "w_slow",
    )

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: getattr(self, k).copy() for k in self._STATE_ARRAYS}
        out["t"] = self.t
        out["step_count"] = self.step_count
        out["rng"] = self.rng.bit_generator.state
        out["noise_buf"] = self._noise_buf.copy()
        out["noise_pos"] = self._noise_pos
        out["rules"] = {
            p.name: p.rule.state_dict() for p in self._plastic if hasattr(p.rule, "state_dict")
        }
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self.built:
            self.build()
        for k in self._STATE_ARRAYS:
            arr = np.asarray(state[k])
            if arr.shape != getattr(self, k).shape:
                raise ValueError(f"state {k!r} has shape {arr.shape}, expected {getattr(self, k).shape}")
            getattr(self, k)[...] = arr
        self.t = float(state["t"])
        self.step_count = int(state["step_count"])
        self.rng.bit_generator.state = state["rng"]
        self._noise_buf = np.array(state["noise_buf"])
        self._noise_pos = int(state["noise_pos"])
        for p in self._plastic:
            if p.name in state.get("rules", {}) and hasattr(p.rule, "load_state_dict"):
                p.rule.load_state_dict(state["rules"][p.name])


def iter_channels(pop: Population) -> Iterable[slice]:
    for k in range(pop.channels):
        yield pop.channel_slice(k)
