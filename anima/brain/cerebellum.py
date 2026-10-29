"""Cerebellum: a forward model that predicts the next moment of sensation.

The cerebellum learns internal models with an error-driven rule: climbing
fibres deliver the error, parallel-fibre synapses change in proportion to
(error x presynaptic activity) — the delta / LMS rule, which is local
(Marr 1969, Albus 1971; Wolpert, Miall & Kawato 1998).

Here it predicts the exteroceptive input ~100 ms ahead from the current
input, bodily state and the motor/vocal commands just issued (efference
copy).  Its prediction error is used twice:

* surprise -> noradrenaline burst and memory salience;
* novelty  -> relieves boredom (curiosity is the drive to find things the
  forward model cannot yet predict).

Because the agent's own voice is predicted from the efference copy, hearing
itself is not surprising while hearing *you* is (von Holst's reafference
principle).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CerebellumConfig:
    lr: float = 0.02
    horizon_ticks: int = 10  # predict this many brain ticks ahead (10 ms each)
    error_tau_ticks: float = 20.0


class Cerebellum:
    def __init__(self, n_in: int, n_out: int, config: CerebellumConfig | None = None) -> None:
        self.cfg = config or CerebellumConfig()
        self.n_in, self.n_out = n_in, n_out
        self.W = np.zeros((n_out, n_in + 1))
        self._hist: deque[np.ndarray] = deque(maxlen=self.cfg.horizon_ticks)
        self.error = 0.0
        self.surprise = 0.0
        self.learning = True

    def step(self, x: np.ndarray, y_now: np.ndarray) -> float:
        """Feed the current input ``x`` and current sensation ``y_now``.

        Learns to predict ``y_now`` from the input seen ``horizon`` ticks ago
        and returns the current (smoothed) surprise in [0, 1].
        """
        x = np.append(np.asarray(x, dtype=float), 1.0)
        if len(self._hist) == self._hist.maxlen:
            x_old = self._hist[0]
            pred = self.W @ x_old
            err = y_now - pred
            if self.learning:
                self.W += self.cfg.lr * np.outer(err, x_old) / (1.0 + x_old @ x_old) * 4.0
            e = float(np.sqrt(np.mean(err**2)))
            a = 1.0 / self.cfg.error_tau_ticks
            self.error += (e - self.error) * a
            # surprise: instantaneous error relative to its recent average
            self.surprise = float(np.clip(2.5 * e, 0.0, 1.0))
        self._hist.append(x)
        return self.surprise

    def state_dict(self) -> dict:
        return {"W": self.W.copy(), "error": self.error, "surprise": self.surprise,
                "hist": [h.copy() for h in self._hist]}

    def load_state_dict(self, s: dict) -> None:
        self.W[...] = s["W"]
        self.error = float(s["error"])
        self.surprise = float(s["surprise"])
        self._hist.clear()
        for h in s.get("hist", []):
            self._hist.append(np.asarray(h, dtype=float))
