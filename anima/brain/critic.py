"""Ventral striatum: learns how good the current situation is.

Dopamine neurons do not report reward; they report *reward prediction
error* — reward plus the change in expected future reward (Schultz, Dayan &
Montague 1997).  The expectation is learned by the ventral striatum.  This
module is that critic: a linear value function V(s) over the brain's current
sensory and interoceptive inputs, learned with TD(lambda):

    delta = r + gamma * V(s_t) - V(s_{t-1})        (-> phasic dopamine)
    w    += alpha * delta * e,   e = gamma * lambda * e + phi(s_{t-1})

Two consequences matter for learning:

* the baseline is *state-specific*, so an action is judged against what is
  usually achievable in that situation (lower-variance credit than a global
  average);
* dopamine shifts from the reward to whatever predicts it — once the creature
  knows that facing food leads to eating, turning toward food is rewarding
  in itself.

It is an abstract (non-spiking) model of the structure; the dopamine it
produces is consumed by spiking synapses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CriticConfig:
    tau_horizon_s: float = 1.5  # discounting timescale (gamma per tick from this)
    tau_trace_s: float = 0.3  # TD(lambda) eligibility timescale
    lr: float = 0.05
    enabled: bool = True


class Critic:
    def __init__(self, n_features: int, tick_s: float = 0.01, config: CriticConfig | None = None) -> None:
        self.cfg = config or CriticConfig()
        self.n = n_features
        self.w = np.zeros(n_features + 1)
        self.e = np.zeros(n_features + 1)
        self.set_tick(tick_s)
        self._phi_prev: np.ndarray | None = None
        self.value = 0.0
        self.delta = 0.0

    def set_tick(self, tick_s: float) -> None:
        self.tick_s = tick_s
        self.gamma = float(np.exp(-tick_s / self.cfg.tau_horizon_s))
        self.lam = float(np.exp(-tick_s / self.cfg.tau_trace_s)) / self.gamma

    def update(self, features: np.ndarray, reward: float, learn: bool = True) -> float:
        """Advance one tick; return the TD error for this tick."""
        phi = np.append(np.asarray(features, dtype=float), 1.0)
        v = float(self.w @ phi)
        if self._phi_prev is None:
            self._phi_prev = phi
            self.value = v
            self.delta = reward
            return reward
        v_prev = float(self.w @ self._phi_prev)
        delta = reward + self.gamma * v - v_prev
        self.e = self.gamma * self.lam * self.e + self._phi_prev
        if learn:
            self.w += self.cfg.lr * delta * self.e / (1.0 + float(self.e @ self.e))
        self._phi_prev = phi
        self.value = v
        self.delta = delta
        return delta

    def reset_episode(self) -> None:
        self.e[:] = 0.0
        self._phi_prev = None

    def state_dict(self) -> dict:
        return {"w": self.w.copy(), "e": self.e.copy(), "value": self.value, "delta": self.delta,
                "cfg": asdict(self.cfg),
                "phi_prev": None if self._phi_prev is None else self._phi_prev.copy()}

    def load_state_dict(self, s: dict) -> None:
        self.cfg = CriticConfig(**s["cfg"])
        self.w[...] = s["w"]
        self.e[...] = s["e"]
        self.value = float(s["value"])
        self.delta = float(s["delta"])
        self.set_tick(self.tick_s)
        prev = s.get("phi_prev")
        self._phi_prev = None if prev is None else np.asarray(prev, dtype=float)
