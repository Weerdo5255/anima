"""Neuron models.

anima uses the Izhikevich (2003) spiking neuron:

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    if v >= 30 mV:  v <- c,  u <- u + d

Two variables are enough to reproduce most cortical firing patterns
(regular spiking, bursting, chattering, fast spiking, ...) at a cost of a
few floating point operations per neuron per millisecond, which is what
makes a continuously running brain feasible on a laptop.

Reference: E. M. Izhikevich, "Simple model of spiking neurons",
IEEE Trans. Neural Networks 14(6), 2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IzhikevichParams:
    """Parameters of one Izhikevich cell type."""

    a: float
    b: float
    c: float
    d: float
    description: str = ""


#: Named cell types.  Excitatory cells follow Dale's law in the network and
#: only ever make excitatory synapses; inhibitory ones only inhibitory.
PRESETS: dict[str, IzhikevichParams] = {
    "RS": IzhikevichParams(0.02, 0.20, -65.0, 8.0, "regular spiking pyramidal cell"),
    "IB": IzhikevichParams(0.02, 0.20, -55.0, 4.0, "intrinsically bursting pyramidal cell"),
    "CH": IzhikevichParams(0.02, 0.20, -50.0, 2.0, "chattering (fast rhythmic bursting) cell"),
    "FS": IzhikevichParams(0.10, 0.20, -65.0, 2.0, "fast spiking interneuron (basket cell)"),
    "LTS": IzhikevichParams(0.02, 0.25, -65.0, 2.0, "low-threshold spiking interneuron"),
    "TC": IzhikevichParams(0.02, 0.25, -65.0, 0.05, "thalamo-cortical relay cell"),
    "RZ": IzhikevichParams(0.10, 0.26, -65.0, 2.0, "resonator"),
    # Medium spiny neurons sit far below threshold at rest; we approximate
    # them with a strongly adapting regular-spiking cell plus negative bias
    # (set by the population, not here).
    "MSN": IzhikevichParams(0.02, 0.20, -65.0, 8.0, "striatal medium spiny neuron (approx.)"),
}

V_PEAK = 30.0


def make_params(
    kind: str, n: int, rng: np.random.Generator, jitter: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-neuron (a, b, c, d) arrays for ``n`` cells of type ``kind``.

    ``jitter`` in [0, 1] adds the cell-to-cell heterogeneity used in
    Izhikevich (2003): excitatory cells drift towards bursting / chattering
    (c up, d down), inhibitory cells towards faster dynamics.  Real cortex is
    heterogeneous and heterogeneity desynchronises networks, which keeps
    spontaneous activity irregular instead of epileptic.
    """
    if kind not in PRESETS:
        raise KeyError(f"unknown neuron type {kind!r}; choose from {sorted(PRESETS)}")
    p = PRESETS[kind]
    a = np.full(n, p.a)
    b = np.full(n, p.b)
    c = np.full(n, p.c)
    d = np.full(n, p.d)
    if jitter > 0:
        r = rng.random(n) * jitter
        if kind in ("FS", "LTS"):
            a = a + 0.08 * r
            b = b - 0.05 * r
        else:
            c = c + 15.0 * r**2
            d = d - 6.0 * r**2
    return a, b, c, d
