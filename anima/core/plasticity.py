"""Local learning rules.

Nothing in anima is trained by back-propagation.  Every weight change is
computed from quantities available *at the synapse*: the timing of the pre-
and postsynaptic spikes (traces), the current weight, and globally broadcast
neuromodulators (dopamine, acetylcholine), exactly the information a real
synapse has.

Rules
-----
STDP
    Spike-timing-dependent plasticity (Bi & Poo 1998): pre-before-post
    potentiates, post-before-pre depresses.  Unsupervised; builds
    sequence/association structure in cortex.
RewardSTDP
    Three-factor rule (Izhikevich 2007, "Solving the distal reward problem"):
    STDP does not change the weight directly but charges a slowly decaying
    *eligibility trace*; the weight only changes when dopamine arrives.
    This lets a reward that comes hundreds of milliseconds later reinforce the
    synapses that caused the rewarded behaviour.
Hebbian
    Symmetric coincidence learning ("cells that fire together wire
    together") with postsynaptic normalisation (von der Malsburg 1973), which
    makes inputs compete.  Used for sensorimotor (mirror) mappings.

All amplitudes are expressed as fractions of a projection's ``w_max`` so the
same rule parameters work across projections with different weight scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .network import Network, Projection


class Rule:
    """Base class.  A rule instance belongs to exactly one projection."""

    def setup(self, net: "Network", proj: "Projection") -> None:
        pass

    def on_spikes(self, net: "Network", proj: "Projection",
                  pre: np.ndarray | None, post: np.ndarray | None) -> None:
        """Called on every step in which a pre- or postsynaptic neuron fired.

        ``pre``/``post`` are indices *local* to the projection's block.
        """

    def on_slow(self, net: "Network", proj: "Projection", dt_ms: float) -> None:
        """Called every ``net.slow_every`` ms."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    def describe(self) -> str:
        return type(self).__name__


def _gate(net: "Network", name: str | None) -> float:
    return 1.0 if name is None else float(net.modulators.get(name, 1.0))


class STDP(Rule):
    """Pair-based STDP with exponential traces (tau = 20 ms).

    Parameters
    ----------
    a_plus, a_minus:
        Potentiation / depression amplitude per spike pairing, as a fraction
        of ``w_max``.  Slightly stronger depression keeps the rule stable.
    soft_bounds:
        Multiplicative (weight-dependent) updates; produce unimodal, stable
        weight distributions instead of all-or-none ones.
    gate:
        Optional neuromodulator name whose level scales learning (e.g.
        ``"acetylcholine"`` — learning is stronger in attentive waking).
    """

    def __init__(self, a_plus: float = 0.005, a_minus: float = 0.006,
                 soft_bounds: bool = True, gate: str | None = None,
                 sparse: bool | None = None) -> None:
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.soft_bounds = soft_bounds
        self.gate = gate
        #: update only existing synapses via index lists (fast for sparse,
        #: bursty recurrent connectivity); None = decide from the density
        self.sparse = sparse

    def setup(self, net, proj) -> None:
        M = net.M[proj.pre_slice, proj.post_slice]
        self.mask_f = M.astype(np.float32)
        density = float(M.mean()) if M.size else 0.0
        self._use_sparse = (density < 0.3) if self.sparse is None else bool(self.sparse)
        if not self._use_sparse:
            return
        N = net.N
        p0, q0 = proj.pre_pop.start, proj.post_pop.start
        rows, cols = np.nonzero(M)  # row-major order
        flat = ((p0 + rows) * N + (q0 + cols)).astype(np.int64)
        # by presynaptic neuron (for LTD on pre spikes)
        r_ptr = np.searchsorted(rows, np.arange(M.shape[0] + 1))
        self._r_cols = [cols[r_ptr[i]:r_ptr[i + 1]] + q0 for i in range(M.shape[0])]
        self._r_flat = [flat[r_ptr[i]:r_ptr[i + 1]] for i in range(M.shape[0])]
        # by postsynaptic neuron (for LTP on post spikes)
        order = np.lexsort((rows, cols))
        c_rows, c_cols, c_flat = rows[order], cols[order], flat[order]
        c_ptr = np.searchsorted(c_cols, np.arange(M.shape[1] + 1))
        self._c_rows = [c_rows[c_ptr[j]:c_ptr[j + 1]] + p0 for j in range(M.shape[1])]
        self._c_flat = [c_flat[c_ptr[j]:c_ptr[j + 1]] for j in range(M.shape[1])]

    def _on_spikes_sparse(self, net, proj, pre, post, g: float) -> None:
        Wf = net.W.reshape(-1)  # view (W is C-contiguous)
        wmax = np.float32(proj.w_max)
        if post is not None and self.a_plus:
            if len(post) == 1:
                rows, flat = self._c_rows[post[0]], self._c_flat[post[0]]
            else:
                rows = np.concatenate([self._c_rows[j] for j in post])
                flat = np.concatenate([self._c_flat[j] for j in post])
            if flat.size:
                x = net.trace[rows].astype(np.float32)
                w = Wf[flat]
                if self.soft_bounds:
                    w += np.float32(g * self.a_plus) * x * (wmax - w)
                else:
                    w += np.float32(g * self.a_plus) * wmax * x
                    np.clip(w, 0.0, wmax, out=w)
                Wf[flat] = w
        if pre is not None and self.a_minus:
            if len(pre) == 1:
                cols, flat = self._r_cols[pre[0]], self._r_flat[pre[0]]
            else:
                cols = np.concatenate([self._r_cols[i] for i in pre])
                flat = np.concatenate([self._r_flat[i] for i in pre])
            if flat.size:
                y = net.trace[cols].astype(np.float32)
                w = Wf[flat]
                if self.soft_bounds:
                    w -= np.float32(g * self.a_minus) * y * w
                else:
                    w -= np.float32(g * self.a_minus) * wmax * y
                    np.clip(w, 0.0, wmax, out=w)
                Wf[flat] = w

    def on_spikes(self, net, proj, pre, post) -> None:
        g = _gate(net, self.gate)
        if g <= 0:
            return
        if self._use_sparse:
            self._on_spikes_sparse(net, proj, pre, post, g)
            return
        W = net.W[proj.pre_slice, proj.post_slice]
        Mf = self.mask_f
        wmax = np.float32(proj.w_max)
        if post is not None and self.a_plus:
            x = net.trace[proj.pre_slice].astype(np.float32)[:, None]
            cur = W[:, post]
            if self.soft_bounds:
                # dw = A+ x (w_max - w): never exceeds w_max, no clipping needed
                cur += np.float32(g * self.a_plus) * x * (wmax - cur) * Mf[:, post]
            else:
                cur += np.float32(g * self.a_plus * wmax) * x * Mf[:, post]
                np.clip(cur, 0.0, wmax, out=cur)
            W[:, post] = cur
        if pre is not None and self.a_minus:
            y = net.trace[proj.post_slice].astype(np.float32)[None, :]
            cur = W[pre, :]
            if self.soft_bounds:
                cur -= np.float32(g * self.a_minus) * y * cur  # dw = -A- y w; stays >= 0
            else:
                cur -= np.float32(g * self.a_minus * wmax) * y * Mf[pre, :]
                np.clip(cur, 0.0, wmax, out=cur)
            W[pre, :] = cur

    def describe(self) -> str:
        return f"STDP(+{self.a_plus}, -{self.a_minus})"


class RewardSTDP(Rule):
    """Dopamine-gated STDP with eligibility traces.

    weight change  dw/dt = lr * w_max * DA(t) * e(t)
    eligibility    de/dt = -e / tau_e + STDP(pre, post)

    ``DA`` is the *phasic* (signed, baseline-subtracted) dopamine level
    published by the brain in ``net.modulators["dopamine"]``.

    The eligibility matrix decays every millisecond in principle; we avoid
    touching the whole matrix each step by storing it in a scaled frame
    (all entries share one decay constant), renormalising occasionally.
    """

    def __init__(self, a_plus: float = 1.0, a_minus: float = 1.0, tau_e: float = 1000.0,
                 lr: float = 0.05, modulator: str = "dopamine", w_min: float = 0.0,
                 apply_every: float = 50.0, normalize: str | None = None) -> None:
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_e = tau_e
        self.lr = lr
        self.modulator = modulator
        self.w_min = w_min
        #: weight updates are applied in batches every ``apply_every`` ms with
        #: dopamine integrated over the batch (exact in the limit
        #: apply_every << tau_e; saves touching the whole matrix every 10 ms)
        self.apply_every = apply_every
        #: Heterosynaptic competition for a fixed synaptic budget:
        #: "pre"  - each input neuron's total output weight is conserved, so
        #:          strengthening its link to one action weakens the others;
        #: "post" - each target neuron's total input is conserved, so no
        #:          action can gain drive from *every* input and take over;
        #: "both" - alternate the two.  None - no competition.
        if normalize not in (None, "pre", "post", "both"):
            raise ValueError(f"normalize must be None, 'pre', 'post' or 'both', not {normalize!r}")
        self.normalize = normalize

    def setup(self, net, proj) -> None:
        shape = (proj.pre_pop.size, proj.post_pop.size)
        self.E = np.zeros(shape, dtype=np.float32)
        self.t_ref = net.t
        self.mask_f = net.M[proj.pre_slice, proj.post_slice].astype(np.float32)
        W = net.W[proj.pre_slice, proj.post_slice]
        self.budget_pre = W.sum(axis=1).astype(np.float32)
        self.budget_post = W.sum(axis=0).astype(np.float32)
        self._da_int = 0.0
        self._since = 0.0

    def _normalize(self, W: np.ndarray, w_max: float) -> None:
        eps = np.float32(1e-12)
        if self.normalize in ("post", "both"):
            W *= (self.budget_post / np.maximum(W.sum(axis=0), eps))[None, :]
        if self.normalize in ("pre", "both"):
            W *= (self.budget_pre / np.maximum(W.sum(axis=1), eps))[:, None]
        np.minimum(W, np.float32(w_max), out=W)

    def _scale(self, t: float) -> float:
        return float(np.exp((t - self.t_ref) / self.tau_e))

    def _renormalize(self, t: float) -> None:
        """Move the stored traces back to an unscaled frame (keeps numbers small,
        also after plasticity was paused for a long time)."""
        if t - self.t_ref > 2.0 * self.tau_e:
            self.E *= np.float32(np.exp(-(t - self.t_ref) / self.tau_e))
            self.t_ref = t

    def eligibility(self, net) -> np.ndarray:
        """Current (true-frame) eligibility trace matrix."""
        self._renormalize(net.t)
        return self.E / self._scale(net.t)

    def on_spikes(self, net, proj, pre, post) -> None:
        self._renormalize(net.t)
        s = self._scale(net.t)
        if post is not None and self.a_plus:
            x = net.trace[proj.pre_slice].astype(np.float32)[:, None]
            self.E[:, post] += np.float32(self.a_plus * s) * x * self.mask_f[:, post]
        if pre is not None and self.a_minus:
            y = net.trace[proj.post_slice].astype(np.float32)[None, :]
            self.E[pre, :] -= np.float32(self.a_minus * s) * y * self.mask_f[pre, :]

    def on_slow(self, net, proj, dt_ms) -> None:
        self._da_int += float(net.modulators.get(self.modulator, 0.0)) * dt_ms / 1000.0
        self._since += dt_ms
        if self._since < self.apply_every:
            return
        self._since = 0.0
        self._renormalize(net.t)
        da_int, self._da_int = self._da_int, 0.0
        if abs(da_int) < 1e-7:
            return
        W = net.W[proj.pre_slice, proj.post_slice]
        k = self.lr * proj.w_max * da_int / self._scale(net.t)
        W += np.float32(k) * self.E
        np.minimum(W, np.float32(proj.w_max), out=W)
        if self.w_min > 0:
            # keep existing synapses from disappearing entirely
            np.maximum(W, np.float32(self.w_min) * self.mask_f, out=W)
        else:
            np.maximum(W, np.float32(0.0), out=W)
        if self.normalize:
            self._normalize(W, proj.w_max)

    def state_dict(self) -> dict[str, Any]:
        return {"E": self.E.copy(), "t_ref": self.t_ref, "da_int": self._da_int,
                "since": self._since, "budget_pre": self.budget_pre.copy(),
                "budget_post": self.budget_post.copy()}

    def load_state_dict(self, state) -> None:
        self.E[...] = state["E"]
        self.t_ref = float(state["t_ref"])
        self._da_int = float(state.get("da_int", 0.0))
        self._since = float(state.get("since", 0.0))
        if "budget_pre" in state:
            self.budget_pre[...] = state["budget_pre"]
            self.budget_post[...] = state["budget_post"]

    def describe(self) -> str:
        return f"RewardSTDP(tau_e={self.tau_e:g}ms, lr={self.lr:g})"


class Hebbian(Rule):
    """Symmetric coincidence learning with postsynaptic normalisation.

    Every pre/post coincidence within ~20 ms potentiates, regardless of
    order.  Every ``norm_every`` ms each postsynaptic neuron's total input
    weight from this projection is rescaled back to its initial total, so
    inputs compete for a fixed synaptic budget: consistently co-active inputs
    win, inconsistent ones fade.
    """

    def __init__(self, rate: float = 0.002, norm_every: float = 100.0,
                 gate: str | None = None) -> None:
        self.rate = rate
        self.norm_every = norm_every
        self.gate = gate

    def setup(self, net, proj) -> None:
        W = net.W[proj.pre_slice, proj.post_slice]
        self.budget = W.sum(axis=0).astype(np.float64)
        self.mask_f = net.M[proj.pre_slice, proj.post_slice].astype(np.float32)
        self._since = 0.0

    def on_spikes(self, net, proj, pre, post) -> None:
        g = _gate(net, self.gate)
        if g <= 0:
            return
        W = net.W[proj.pre_slice, proj.post_slice]
        k = np.float32(g * self.rate * proj.w_max)
        if post is not None:
            x = net.trace[proj.pre_slice].astype(np.float32)[:, None]
            W[:, post] += k * x * self.mask_f[:, post]
        if pre is not None:
            y = net.trace[proj.post_slice].astype(np.float32)[None, :]
            W[pre, :] += k * y * self.mask_f[pre, :]

    def on_slow(self, net, proj, dt_ms) -> None:
        self._since += dt_ms
        if self._since < self.norm_every:
            return
        self._since = 0.0
        W = net.W[proj.pre_slice, proj.post_slice]
        np.minimum(W, proj.w_max, out=W)
        total = W.sum(axis=0)
        ok = total > 0
        factor = np.ones_like(total)
        factor[ok] = self.budget[ok] / total[ok]
        W *= factor[None, :].astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"budget": self.budget.copy(), "since": self._since}

    def load_state_dict(self, state) -> None:
        self.budget[...] = state["budget"]
        self._since = float(state.get("since", 0.0))

    def describe(self) -> str:
        return f"Hebbian(rate={self.rate:g})"


RULES = {"stdp": STDP, "reward_stdp": RewardSTDP, "hebbian": Hebbian}


def make_rule(spec: Any) -> Rule | None:
    """Create a rule from a spec: ``None``, a Rule, a name, or ``{"type": name, ...}``."""
    if spec is None or isinstance(spec, Rule):
        return spec
    if isinstance(spec, str):
        return RULES[spec]()
    if isinstance(spec, dict):
        spec = dict(spec)
        kind = spec.pop("type")
        return RULES[kind](**spec)
    raise TypeError(f"cannot make a plasticity rule from {spec!r}")
