"""The device engine: a network's dynamics on a GPU.

:class:`~anima.core.network.Network` runs on the CPU with NumPy unless it is
given another ``device``; then it hands its arrays to a :class:`DeviceEngine`,
which keeps them on that device and simulates the same model:

* the same Izhikevich neurons, conductance synapses, adaptation, traces and
  homeostasis, step for step, every millisecond;
* the same learning rules (STDP, dopamine-gated STDP with eligibility
  traces, Hebbian) with the same parameters, read from the rule objects;
* creatures save to the same files, so a creature raised on a GPU can live
  on a CPU and back.

What is different, to keep a GPU busy instead of waiting:

* **numbers are float32** (the reference engine uses float64 for the
  neuron state; weights are float32 in both);
* **plasticity is applied once per 10 ms window** from the window's spike
  trains, with small matrix products, instead of at every spike.  For the
  eligibility traces of reward learning this is exact.  A Hebbian or STDP
  weight change takes effect at the end of its 10 ms window instead of
  immediately (and the soft bound of STDP sees the weight at the start of
  the window — a second-order difference);
* **background noise is drawn on the device** (``noise="host"`` draws it
  from the same NumPy stream as the reference engine, for checking);
* per millisecond the host waits for the device once at most (to learn
  which neurons fired, for event-driven spike propagation); small networks
  skip even that with a dense matrix-vector product (``propagation``).

Per 10 ms tick the brain, body and thought log need a few small readouts
(activity traces, firing rates); they are copied to the host once per tick.
"""

from __future__ import annotations

import math
import threading
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from .device import DeviceSpec, make_backend
from .neurons import V_PEAK
from .plasticity import STDP, Hebbian, RewardSTDP

if TYPE_CHECKING:  # pragma: no cover
    from .network import Network, Projection

E_GABA = -75.0
#: networks up to this many neurons propagate spikes with one dense
#: matrix-vector product per millisecond on a GPU (it reads all of W, but
#: needs no round trip to the host); bigger ones send only what fired
DENSE_MAX_N = 2500
#: names of the per-neuron settings the brain writes every tick (host -> device)
PARAM_NAMES = ("I_ext", "gain", "adapt", "bias_mod", "noise_sigma", "slow_inc")
#: per-neuron state kept on the device, with the row of the stacked state array
S_ROWS = {"g_ampa": 0, "g_gaba": 1, "g_nmda": 2, "trace": 3, "readout": 4, "rate": 5, "w_slow": 6}


class DeviceEngine:
    """Runs a built :class:`Network` on a device.  Created by the network."""

    def __init__(self, net: "Network", spec: DeviceSpec, dtype: Any = np.float32, noise: str = "device",
                 propagation: str = "auto", fused: bool = True) -> None:
        if noise not in ("device", "host"):
            raise ValueError("noise must be 'device' or 'host'")
        if propagation not in ("auto", "dense", "sparse"):
            raise ValueError("propagation must be 'auto', 'dense' or 'sparse'")
        self.net = net
        self.spec = spec
        self.B = B = make_backend(spec, dtype)
        self.noise_mode = noise
        N = net.N
        if propagation == "auto":
            propagation = "dense" if (B.name != "numpy" and N <= DENSE_MAX_N) else "sparse"
        self.propagation = propagation
        f = B.float
        # ---- constants
        self.a = B.asarray(net.a, f)
        self.b = B.asarray(net.b, f)
        self.c = B.asarray(net.c, f)
        self.d = B.asarray(net.d_base, f)
        self.is_inh_host = net.is_inh.copy()
        self.exc_f = B.asarray((~net.is_inh).astype(np.float64), f)
        self.inh_f = B.asarray(net.is_inh.astype(np.float64), f)
        self.bias_base = B.asarray(net.bias_base, f)
        self.nmda_ratio = B.asarray(net.nmda_ratio, f)
        self.has_nmda = bool(net.has_nmda)
        has_target = ~np.isnan(net.target_rate)
        self.any_target = bool(has_target.any())
        self.has_target = B.asarray(has_target, B.bool)
        self.target = B.asarray(np.nan_to_num(net.target_rate, nan=0.0), f)
        self.decay = B.asarray(net._decay, f)  # (7, 1)
        self._decay_host = [float(x) for x in net._decay[:, 0]]
        from . import network as _netmod
        self.rate_inc = 1000.0 / _netmod.TAU_RATE
        self.exc32 = B.astype(self.exc_f, np.float32)
        self.inh32 = B.astype(self.inh_f, np.float32)
        self.EI = B.stack([self.exc32, self.inh32])  # (2, N): who is excitatory, who inhibitory
        # ---- state
        self.v = B.asarray(net.v, f)
        self.u = B.asarray(net.u, f)
        self.S = B.asarray(net._S, f)
        self.bias_homeo = B.asarray(net.bias_homeo, f)
        self.spike_count = B.asarray(net.spike_count, B.int)
        self.W = B.asarray(net.W, np.float32)
        self.fired = B.zeros(N, B.bool)
        # ---- the brain's per-neuron settings, uploaded when they change
        self.P = B.zeros((len(PARAM_NAMES), N), f)
        self._params_dirty = True
        # ---- plasticity: per-rule device state, and the window's spike trains
        T = net.slow_every
        self.T = T
        self.X_rec = B.zeros((T, N), f)  # traces each step, before that step's spikes
        self.S_rec = B.zeros((T, N), f)  # spikes each step
        self._window_used = False
        self.rules: dict[str, dict[str, Any]] = {}
        for proj in net._plastic:
            self.rules[proj.name] = self._rule_setup(proj)
        self._scaling = [p for p in net.projections.values() if p.scaling]
        # ---- noise
        self._noise_buf = None
        self._noise_pos = 0
        self._gen = None
        self.reseed_noise(_seed_from_state(net.rng.bit_generator.state))
        # ---- host caches (valid until the next step or change)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_key: tuple[int, int] = (-1, -1)
        self.version = 0
        self._noise_src = None
        self._thread = threading.get_ident()
        # ---- the fused per-neuron kernel (checked against the array path)
        self.kernel = None
        if fused:
            self.kernel = self._checked_kernel()

    # ================================================================ setup
    def _rule_setup(self, proj: "Projection") -> dict[str, Any]:
        B, net = self.B, self.net
        P, Q = proj.pre_slice, proj.post_slice
        M = net.M[P, Q]
        st: dict[str, Any] = {"mask": B.asarray(M, B.bool)}  # (1 byte per synapse slot)
        rule = proj.rule
        if isinstance(rule, RewardSTDP):
            Wb = net.W[P, Q]
            st.update(E=B.zeros(M.shape, np.float32), t_ref=float(net.t), da_int=0.0, since=0.0,
                      budget_pre=B.asarray(Wb.sum(axis=1).astype(np.float32), np.float32),
                      budget_post=B.asarray(Wb.sum(axis=0).astype(np.float32), np.float32))
        elif isinstance(rule, Hebbian):
            st.update(budget=B.asarray(net.W[P, Q].sum(axis=0).astype(np.float64), self.B.float), since=0.0)
        elif not isinstance(rule, STDP):
            raise TypeError(f"{proj.name}: the device engine knows STDP, RewardSTDP and Hebbian, "
                            f"not {type(rule).__name__}")
        return st

    def reseed_noise(self, seed: int) -> None:
        """New background-noise stream on the device (``noise="device"``)."""
        self._gen = self.B.rng(seed)
        self._noise_buf = None
        self._noise_pos = 0

    # ================================================================ living
    def mark_params(self) -> None:
        self._params_dirty = True

    def _upload_params(self) -> None:
        net = self.net
        host = np.stack([np.asarray(getattr(net, n), dtype=np.float64) for n in PARAM_NAMES])
        self.P[...] = self.B.asarray(host, self.B.float)
        self._params_dirty = False

    def _noise_row(self) -> Any:
        net = self.net
        if self.noise_mode == "host":  # the reference engine's stream, exactly
            if net._noise_pos >= len(net._noise_buf):
                net._noise_buf = net.rng.standard_normal((200, net.N))
                net._noise_pos = 0
            if self._noise_src is not net._noise_buf:
                self._noise_buf = self.B.asarray(net._noise_buf, self.B.float)
                self._noise_src = net._noise_buf
            row = self._noise_buf[net._noise_pos]
            net._noise_pos += 1
            return row
        if self._noise_buf is None or self._noise_pos >= self._noise_buf.shape[0]:
            self._noise_buf = self.B.standard_normal(self._gen, (200, net.N))
            self._noise_pos = 0
        row = self._noise_buf[self._noise_pos]
        self._noise_pos += 1
        return row

    def step(self) -> None:
        net = self.net
        if self._thread != threading.get_ident():  # (CuPy keeps a current device per thread)
            self.B.activate()
            self._thread = threading.get_ident()
        k = net.step_count % self.T
        if self._params_dirty or k == 0:
            self._upload_params()
        noise = self._noise_row()
        if self.kernel is not None:
            self._neurons_fused(noise, k)
        else:
            self._neurons(noise, k)
        self._propagate(k)
        self._window_used = True
        net.t += net.dt
        net.step_count += 1
        if net.step_count % self.T == 0:
            self.end_window()
        if net._post_step_hooks:
            idx = self.B.to_host(self.B.nonzero(self.fired))
            for hook in net._post_step_hooks:
                hook(net, idx)

    def run(self, ms: float) -> None:
        for _ in range(int(round(ms / self.net.dt))):
            self.step()

    def _neurons(self, noise: Any, k: int) -> None:
        """One millisecond of every neuron (the array path; the reference
        engine's arithmetic, in the same order)."""
        B, net = self.B, self.net
        v, u, S, P = self.v, self.u, self.S, self.P
        g_ampa, g_gaba, g_nmda = S[0], S[1], S[2]
        G = g_ampa + g_gaba
        GE = g_gaba * E_GABA
        if self.has_nmda:
            mg = B.exp(v * -0.062)
            mg *= 0.28
            mg += 1.0
            G += g_nmda / mg
        G *= P[1]  # gain
        GE *= P[1]
        G += 1e-9
        G *= -0.5 * net.dt
        em = B.expm1(G)
        GE *= em
        GE /= G
        GE *= 0.5 * net.dt
        em += 1.0
        J = P[4] * noise  # noise_sigma
        J += P[0]  # I_ext
        J += self.bias_base
        J += P[3]  # bias_mod
        J += self.bias_homeo
        J += 140.0
        J -= S[6]  # w_slow
        J -= u
        J *= 0.5
        for _ in range(2):
            v *= em
            v += GE
            t = v * 0.02
            t += 2.5
            t *= v
            t += J
            v += t
            B.clip_(v, hi=V_PEAK)
        B.clip_(v, lo=-120.0)
        u += self.a * (self.b * v - u)
        fired = v >= V_PEAK
        v[...] = B.where(fired, self.c, v)
        u[...] = B.where(fired, u + self.d * P[2], u)  # adapt
        S *= self.decay
        self.X_rec[k] = S[3]
        ff = B.astype(fired, B.float)
        self.S_rec[k] = ff
        S[3] += ff  # trace
        S[4] += ff  # readout
        S[5] += ff * self.rate_inc  # rate
        S[6] += ff * P[5]  # slow adaptation (slow_inc)
        self.spike_count += B.astype(fired, B.int)
        self.fired = fired

    def _neurons_fused(self, noise: Any, k: int) -> None:
        S, P = self.S, self.P
        d = self._decay_host
        self.kernel(noise, P[0], P[1], P[2], P[3], P[4], P[5], self.bias_base, self.bias_homeo,
                    self.a, self.b, self.c, self.d, d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                    bool(self.has_nmda),
                    self.v, self.u, S[0], S[1], S[2], S[3], S[4], S[5], S[6], self.spike_count,
                    self.fired, self.X_rec[k], self.S_rec[k])

    def _propagate(self, k: int) -> None:
        """Deliver this millisecond's spikes (after the conductances decayed,
        as in the reference engine)."""
        B, S = self.B, self.S
        if self.propagation == "dense":  # no wait: one product with all of W
            ff = self.S_rec[k]  # this step's spikes as 0/1
            if self.S_rec.dtype != self.W.dtype:
                ff = B.astype(ff, np.float32)
            inc = (self.EI * ff) @ self.W
            inc_e, inc_i = inc[0], inc[1]
        else:
            idx = B.nonzero(self.fired)  # the one wait per millisecond
            if int(idx.shape[0]) == 0:
                return
            if B.name == "numpy":  # the reference engine's summation, exactly
                inh = self.is_inh_host[idx]
                exc, inh = idx[~inh], idx[inh]
                inc_e = self.W[exc].sum(axis=0) if exc.size else None
                inc_i = self.W[inh].sum(axis=0) if inh.size else None
            else:
                inc = self.EI[:, idx] @ self.W[idx]
                inc_e, inc_i = inc[0], inc[1]
        if inc_e is not None:
            S[0] += inc_e
            if self.has_nmda:
                S[2] += inc_e * self.nmda_ratio
        if inc_i is not None:
            S[1] += inc_i

    # ================================================================ every 10 ms
    def end_window(self) -> None:
        """Plasticity for the window that just ended, then the slow updates
        (the reference engine's ``_slow_tick``)."""
        net = self.net
        dt_ms = net.slow_every * net.dt
        self.flush_plasticity()
        if net.plasticity_enabled:
            for proj in net._plastic:
                self._on_slow(proj, dt_ms)
        if net.homeostasis_enabled:
            if self.any_target:
                err = self.B.where(self.has_target, self.target - self.S[5], 0.0)
                self.bias_homeo += (net.ip_rate * net.homeo_scale) * err * (dt_ms / 1000.0)
                self.B.clip_(self.bias_homeo, -net.ip_max, net.ip_max)
            if net.step_count % net.scaling_every == 0:
                self._synaptic_scaling()

    def flush_plasticity(self) -> None:
        """Apply the spike-timing plasticity of the steps recorded so far."""
        if self._window_used and self.net.plasticity_enabled and self.rules:
            self._spike_plasticity()
        if self._window_used:
            self.X_rec[...] = 0.0
            self.S_rec[...] = 0.0
        self._window_used = False

    def _window_times(self) -> np.ndarray:
        """Times (ms) at which the recorded steps of the window began."""
        net = self.net
        k = net.step_count % self.T or self.T  # steps recorded in this window
        start = net.step_count - k
        return net.t - (net.step_count - start) * net.dt + np.arange(self.T) * net.dt

    def _spike_plasticity(self) -> None:
        """Spike-timing plasticity for the recorded window: for every plastic
        projection, small matrix products over the window's spike trains and
        traces, restricted to the columns (postsynaptic neurons) and rows
        (presynaptic neurons) that fired — one wait for the device in all."""
        B, net = self.B, self.net
        f32 = np.float32
        spiked = B.nonzero(B.any_axis0(self.S_rec))  # every neuron that fired in the window
        spiked_host = B.to_host(spiked)
        if spiked_host.size == 0:
            return
        times = self._window_times()

        def fired_in(sl: slice) -> Any:
            lo, hi = np.searchsorted(spiked_host, (sl.start, sl.stop))
            return None if hi <= lo else spiked[lo:hi] - sl.start

        for proj in net._plastic:
            rule, st = proj.rule, self.rules[proj.name]
            P, Q = proj.pre_slice, proj.post_slice
            cols, rows = fired_in(Q), fired_in(P)  # local indices: posts and pres that fired
            if cols is None and rows is None:
                continue
            M = st["mask"]
            if isinstance(rule, RewardSTDP):
                # each step's eligibility frame scale: exactly the per-spike rule
                self._renormalize(st, rule, float(times[0]))
                w = np.exp((times - st["t_ref"]) / rule.tau_e).astype(f32)
                wd = B.asarray(w[:, None], f32)
                E = st["E"]
                if rule.a_plus and cols is not None:
                    X = B.astype(self.X_rec[:, P], f32) * wd
                    E[:, cols] += _f(rule.a_plus) * (X.T @ B.astype(self.S_rec[:, Q][:, cols], f32)) * M[:, cols]
                if rule.a_minus and rows is not None:
                    Sp = B.astype(self.S_rec[:, P][:, rows], f32) * wd
                    E[rows, :] -= _f(rule.a_minus) * (Sp.T @ B.astype(self.X_rec[:, Q], f32)) * M[rows, :]
                continue
            g = _gate(net, rule.gate)
            if g <= 0:
                continue
            Wb = self.W[P, Q]
            if isinstance(rule, Hebbian):
                k = _f(g * rule.rate * proj.w_max)
                if cols is not None:
                    X = B.astype(self.X_rec[:, P], f32)
                    Wb[:, cols] += k * (X.T @ B.astype(self.S_rec[:, Q][:, cols], f32)) * M[:, cols]
                if rows is not None:
                    Y = B.astype(self.X_rec[:, Q], f32)
                    Wb[rows, :] += k * (B.astype(self.S_rec[:, P][:, rows], f32).T @ Y) * M[rows, :]
                continue
            wmax = _f(proj.w_max)  # STDP
            if rule.a_plus and cols is not None:
                dp = B.astype(self.X_rec[:, P], f32).T @ B.astype(self.S_rec[:, Q][:, cols], f32)
                blk = Wb[:, cols]
                if rule.soft_bounds:
                    blk += _f(g * rule.a_plus) * dp * (wmax - blk) * M[:, cols]
                else:
                    blk += _f(g * rule.a_plus * proj.w_max) * dp * M[:, cols]
                    B.clip_(blk, 0.0, wmax)
                Wb[:, cols] = blk
            if rule.a_minus and rows is not None:
                dm = B.astype(self.S_rec[:, P][:, rows], f32).T @ B.astype(self.X_rec[:, Q], f32)
                blk = Wb[rows, :]
                if rule.soft_bounds:
                    blk -= _f(g * rule.a_minus) * dm * blk
                else:
                    blk -= _f(g * rule.a_minus * proj.w_max) * dm * M[rows, :]
                    B.clip_(blk, 0.0, wmax)
                Wb[rows, :] = blk

    @staticmethod
    def _renormalize(st: dict[str, Any], rule: RewardSTDP, t: float) -> None:
        if t - st["t_ref"] > 2.0 * rule.tau_e:
            st["E"] *= _f(math.exp(-(t - st["t_ref"]) / rule.tau_e))
            st["t_ref"] = t

    def _on_slow(self, proj: "Projection", dt_ms: float) -> None:
        B, net = self.B, self.net
        rule, st = proj.rule, self.rules[proj.name]
        P, Q = proj.pre_slice, proj.post_slice
        if isinstance(rule, RewardSTDP):
            st["da_int"] += float(net.modulators.get(rule.modulator, 0.0)) * dt_ms / 1000.0
            st["since"] += dt_ms
            if st["since"] < rule.apply_every:
                return
            st["since"] = 0.0
            self._renormalize(st, rule, net.t)
            da_int, st["da_int"] = st["da_int"], 0.0
            if abs(da_int) < 1e-7:
                return
            Wb = self.W[P, Q]
            scale = math.exp((net.t - st["t_ref"]) / rule.tau_e)
            Wb += _f(rule.lr * proj.w_max * da_int / scale) * st["E"]
            B.clip_(Wb, hi=_f(proj.w_max))
            if rule.w_min > 0:
                Wb[...] = B.maximum(Wb, _f(rule.w_min) * st["mask"])
            else:
                B.clip_(Wb, lo=0.0)
            if rule.normalize:
                eps = _f(1e-12)
                if rule.normalize in ("post", "both"):
                    Wb *= (st["budget_post"] / B.maximum(B.sum(Wb, 0), eps))[None, :]
                if rule.normalize in ("pre", "both"):
                    Wb *= (st["budget_pre"] / B.maximum(B.sum(Wb, 1), eps))[:, None]
                B.clip_(Wb, hi=_f(proj.w_max))
        elif isinstance(rule, Hebbian):
            st["since"] += dt_ms
            if st["since"] < rule.norm_every:
                return
            st["since"] = 0.0
            Wb = self.W[P, Q]
            B.clip_(Wb, hi=float(proj.w_max))
            total = B.sum(Wb, 0)
            ok = total > 0
            factor = B.where(ok, st["budget"] / B.where(ok, total, 1.0), 1.0)
            Wb *= B.astype(factor, np.float32)[None, :]

    def _synaptic_scaling(self) -> None:
        net, B = self.net, self.B
        if net.homeo_scale <= 0:
            return
        for proj in self._scaling:
            qs = proj.post_slice
            if np.all(np.isnan(net.target_rate[qs])):
                continue
            rate = self.S[5][qs]
            target = B.where(self.has_target[qs], self.target[qs], rate)
            factor = 1.0 + (net.scaling_rate * net.homeo_scale) * (target - rate) / B.maximum(target, 0.5)
            B.clip_(factor, 0.9, 1.1)
            blk = self.W[proj.pre_slice, qs]
            blk *= B.astype(factor, np.float32)[None, :]
            B.clip_(blk, hi=float(proj.w_max))

    # ================================================================ host views
    def _fresh(self) -> None:
        key = (self.net.step_count, self.version)
        if key != self._cache_key:
            self._cache = {}
            self._cache_key = key

    def host(self, name: str) -> np.ndarray:
        """A read-only host copy of a state array (cached until the next step)."""
        self._fresh()
        if name not in self._cache:
            B = self.B
            if name in ("readout", "rate"):  # the per-tick readouts, fetched together
                both = B.to_host(self.S[4:6]).astype(np.float64)
                for k, n in enumerate(("readout", "rate")):
                    both[k].flags.writeable = False
                    self._cache[n] = both[k]
                return self._cache[name]
            if name in S_ROWS:
                arr = B.to_host(self.S[S_ROWS[name]]).astype(np.float64)
            elif name == "_S":
                arr = B.to_host(self.S).astype(np.float64)
            elif name == "W":
                arr = B.to_host(self.W).astype(np.float32)
            elif name == "last_spikes":
                arr = B.to_host(B.nonzero(self.fired)).astype(np.int64)
            elif name == "spike_count":
                arr = B.to_host(self.spike_count).astype(np.int64)
            elif name == "fired":
                arr = B.to_host(self.fired).astype(bool)
            else:
                arr = B.to_host(getattr(self, name)).astype(np.float64)
            arr.flags.writeable = False
            self._cache[name] = arr
        return self._cache[name]

    def block(self, proj: "Projection") -> np.ndarray:
        return self.B.to_host(self.W[proj.pre_slice, proj.post_slice]).astype(np.float32)

    def set_block(self, proj: "Projection", values: np.ndarray) -> None:
        self.W[proj.pre_slice, proj.post_slice] = self.B.asarray(np.asarray(values, np.float32), np.float32)
        self.version += 1

    def scale_block(self, proj: "Projection", factor: float) -> None:
        blk = self.W[proj.pre_slice, proj.post_slice]
        blk *= _f(factor)
        self.version += 1

    def drive(self, projs: list["Projection"], size: int) -> np.ndarray:
        """Sum over ``projs`` of presynaptic rate x weights, on the device."""
        B = self.B
        total = None
        for p in projs:
            x = B.astype(self.S[5][p.pre_slice], np.float32) @ self.W[p.pre_slice, p.post_slice]
            total = x if total is None else total + x
        return np.zeros(size) if total is None else B.to_host(total).astype(np.float64)

    # ================================================================ saving
    def state_arrays(self) -> dict[str, np.ndarray]:
        """The reference engine's state arrays, as host copies."""
        self.flush_plasticity()  # (agents save on 10 ms boundaries, where nothing is pending)
        B = self.B
        out: dict[str, np.ndarray] = {}
        for k in ("v", "u", "bias_homeo"):
            out[k] = B.to_host(getattr(self, k)).astype(np.float64)
        out["spike_count"] = B.to_host(self.spike_count).astype(np.int64)
        S = B.to_host(self.S).astype(np.float64)
        for name, row in S_ROWS.items():
            out[name] = S[row].copy()
        out["W"] = B.to_host(self.W).astype(np.float32)
        return out

    def load_arrays(self, state: dict[str, Any]) -> None:
        B = self.B
        for k in ("v", "u", "bias_homeo"):
            getattr(self, k)[...] = B.asarray(np.asarray(state[k], np.float64), B.float)
        self.spike_count[...] = B.asarray(np.asarray(state["spike_count"], np.int64), B.int)
        for name, row in S_ROWS.items():
            self.S[row] = B.asarray(np.asarray(state[name], np.float64), B.float)
        self.W[...] = B.asarray(np.asarray(state["W"], np.float32), np.float32)
        self.X_rec[...] = 0.0
        self.S_rec[...] = 0.0
        self._window_used = False
        self._params_dirty = True
        self._noise_src = None
        self.version += 1

    def rule_state(self, proj: "Projection") -> dict[str, Any]:
        """A rule's state in the reference engine's format."""
        st, rule, B = self.rules[proj.name], proj.rule, self.B
        if isinstance(rule, RewardSTDP):
            return {"E": B.to_host(st["E"]).astype(np.float32), "t_ref": st["t_ref"], "da_int": st["da_int"],
                    "since": st["since"], "budget_pre": B.to_host(st["budget_pre"]).astype(np.float32),
                    "budget_post": B.to_host(st["budget_post"]).astype(np.float32)}
        if isinstance(rule, Hebbian):
            return {"budget": B.to_host(st["budget"]).astype(np.float64), "since": st["since"]}
        return {}

    def load_rule_state(self, proj: "Projection", state: dict[str, Any]) -> None:
        st, rule, B = self.rules[proj.name], proj.rule, self.B
        if isinstance(rule, RewardSTDP):
            st["E"][...] = B.asarray(np.asarray(state["E"], np.float32), np.float32)
            st["t_ref"] = float(state["t_ref"])
            st["da_int"] = float(state.get("da_int", 0.0))
            st["since"] = float(state.get("since", 0.0))
            if "budget_pre" in state:
                st["budget_pre"][...] = B.asarray(np.asarray(state["budget_pre"], np.float32), np.float32)
                st["budget_post"][...] = B.asarray(np.asarray(state["budget_post"], np.float32), np.float32)
        elif isinstance(rule, Hebbian):
            st["budget"][...] = B.asarray(np.asarray(state["budget"], np.float64), B.float)
            st["since"] = float(state.get("since", 0.0))
        self.version += 1

    def eligibility(self, proj: "Projection") -> np.ndarray:
        """True-frame eligibility traces of a RewardSTDP projection."""
        st, rule = self.rules[proj.name], proj.rule
        self._renormalize(st, rule, self.net.t)
        return self.B.to_host(st["E"]).astype(np.float32) / math.exp((self.net.t - st["t_ref"]) / rule.tau_e)

    # ================================================================ the fused kernel
    def _checked_kernel(self) -> Any:
        """The backend's fused neuron kernel, if it gives the same numbers as
        the array path on this network (checked on a copy of the state)."""
        try:
            kernel = self.B.neuron_kernel()
        except Exception as exc:  # pragma: no cover - needs a GPU library
            warnings.warn(f"anima: fused neuron kernel unavailable ({exc}); using array operations")
            return None
        if kernel is None:
            return None
        try:  # pragma: no cover - needs a GPU library
            return kernel if self._kernel_agrees(kernel) else None
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"anima: fused neuron kernel failed its check ({exc}); using array operations")
            return None

    def _kernel_agrees(self, kernel: Any) -> bool:  # pragma: no cover - needs a GPU library
        B = self.B
        saved = {k: B.copy(getattr(self, k)) for k in ("v", "u", "S", "spike_count", "X_rec", "S_rec")}
        rng = np.random.default_rng(12345)
        self._upload_params()
        noise = B.asarray(rng.standard_normal(self.net.N) * 3.0, B.float)
        self.v[...] = B.asarray(rng.uniform(-75.0, 35.0, self.net.N), B.float)
        self.S[0] = B.asarray(rng.uniform(0.0, 0.5, self.net.N), B.float)
        self.S[1] = B.asarray(rng.uniform(0.0, 0.5, self.net.N), B.float)
        start = {k: B.copy(getattr(self, k)) for k in ("v", "u", "S", "spike_count")}
        self._neurons(noise, 0)
        ref = {k: B.to_host(getattr(self, k)).astype(np.float64) for k in ("v", "u", "S")}
        ref_fired = B.to_host(self.fired)
        for k, x in start.items():
            getattr(self, k)[...] = x
        self.kernel = kernel
        self._neurons_fused(noise, 0)
        self.kernel = None
        got = {k: B.to_host(getattr(self, k)).astype(np.float64) for k in ("v", "u", "S")}
        ok = bool(np.array_equal(ref_fired, B.to_host(self.fired)))
        for k in ref:
            ok = ok and bool(np.allclose(got[k], ref[k], rtol=1e-4, atol=1e-3))
        for k, x in saved.items():
            getattr(self, k)[...] = x
        self.fired = B.zeros(self.net.N, B.bool)
        if not ok:
            warnings.warn("anima: the fused neuron kernel disagrees with the array path on this device; "
                          "using array operations")
        return ok


def _gate(net: "Network", name: str | None) -> float:
    return 1.0 if name is None else float(net.modulators.get(name, 1.0))


def _f(x: float) -> float:
    """A float32-rounded Python float: what ``np.float32(x)`` gives, usable
    with NumPy, CuPy and PyTorch arrays alike."""
    return float(np.float32(x))


def _seed_from_state(state: dict[str, Any]) -> int:
    """A seed for device noise taken from (not drawn from) a NumPy generator's
    state, so building on a device does not change any other random draw."""
    inner = state.get("state", {})
    s = inner.get("state", 0) if isinstance(inner, dict) else 0
    return int((int(s) ^ 0x616E696D61) % (2 ** 63 - 1))


# ---------------------------------------------------------------- CuPy kernel
_CUPY_NEURON_SRC = r"""
T G = ga + gg;
T GE = gg * (T)(-75.0);
if (has_nmda) {
    T mg = (T)1.0 + (T)0.28 * exp((T)(-0.062) * v);
    G += gn / mg;
}
G *= gain;
GE *= gain;
G += (T)1e-9;
G *= (T)(-0.5);
T em = expm1(G);
GE *= em;
GE /= G;
GE *= (T)0.5;
em += (T)1.0;
T J = sigma * noise + iext + bias_base + bias_mod + bias_homeo + (T)140.0 - ws - u;
J *= (T)0.5;
T vv = v;
for (int h = 0; h < 2; ++h) {
    vv = vv * em + GE;
    T t = ((T)0.02 * vv + (T)2.5) * vv + J;
    vv += t;
    if (vv > (T)30.0) vv = (T)30.0;
}
if (vv < (T)(-120.0)) vv = (T)(-120.0);
T uu = u + a * (b * vv - u);
bool f = vv >= (T)30.0;
if (f) { vv = c; uu += d * adapt; }
v = vv;
u = uu;
ga *= d0; gg *= d1; gn *= d2;
T tr = trace * d3;
rd *= d4; rt *= d5; ws *= d6;
xrec = tr;
if (f) { tr += (T)1.0; rd += (T)1.0; rt += (T)1.0; ws += slow_inc; count += 1; }
trace = tr;
fired = f;
srec = f ? (T)1.0 : (T)0.0;
"""


def cupy_neuron_kernel(cupy: Any, dtype: Any) -> Any:  # pragma: no cover - needs CuPy and a GPU
    """One CUDA kernel for the whole per-neuron update (instead of ~40 array
    operations, each a kernel launch)."""
    kern = cupy.ElementwiseKernel(
        "T noise, T iext, T gain, T adapt, T bias_mod, T sigma, T slow_inc, T bias_base, T bias_homeo, "
        "T a, T b, T c, T d, T d0, T d1, T d2, T d3, T d4, T d5, T d6, bool has_nmda",
        "T v, T u, T ga, T gg, T gn, T trace, T rd, T rt, T ws, int64 count, bool fired, T xrec, T srec",
        _CUPY_NEURON_SRC, "anima_neurons")

    def run(noise, iext, gain, adapt, bias_mod, sigma, slow_inc, bias_base, bias_homeo, a, b, c, d,
            d0, d1, d2, d3, d4, d5, d6, has_nmda, v, u, ga, gg, gn, trace, rd, rt, ws, count, fired,
            xrec, srec):
        t = np.dtype(dtype).type
        kern(noise, iext, gain, adapt, bias_mod, sigma, slow_inc, bias_base, bias_homeo, a, b, c, d,
             t(d0), t(d1), t(d2), t(d3), t(d4), t(d5), t(d6), np.bool_(has_nmda),
             v, u, ga, gg, gn, trace, rd, rt, ws, count, fired, xrec, srec)
    return run


__all__ = ["DeviceEngine", "PARAM_NAMES", "DENSE_MAX_N", "cupy_neuron_kernel"]
