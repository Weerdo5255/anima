"""Recording tools: spike rasters, population rates, scalar probes."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

import numpy as np

from .network import Network


class SpikeRecorder:
    """Keeps the most recent spikes of selected populations (ring buffer)."""

    def __init__(self, net: Network, populations: Iterable[str] | None = None,
                 max_spikes: int = 200_000) -> None:
        self.net = net
        names = list(populations) if populations is not None else list(net.populations)
        self.names = names
        sel = np.zeros(net.N, dtype=bool)
        for n in names:
            sel[net.populations[n].slice] = True
        self._sel = sel
        self.t: deque[float] = deque(maxlen=max_spikes)
        self.i: deque[int] = deque(maxlen=max_spikes)
        net.add_post_step_hook(self._hook)

    def _hook(self, net: Network, idx: np.ndarray) -> None:
        if idx.size:
            keep = idx[self._sel[idx]]
            if keep.size:
                self.t.extend([net.t] * keep.size)
                self.i.extend(keep.tolist())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.fromiter(self.t, float), np.fromiter(self.i, int)

    def clear(self) -> None:
        self.t.clear()
        self.i.clear()


class RateMonitor:
    """Samples the mean firing rate (Hz) of populations every ``every_ms``."""

    def __init__(self, net: Network, populations: Iterable[str] | None = None,
                 every_ms: float = 10.0, max_samples: int = 100_000) -> None:
        self.net = net
        self.names = list(populations) if populations is not None else list(net.populations)
        self.every = int(every_ms)
        self._last = net.spike_count.copy()
        self._t_last = net.t
        self.t: deque[float] = deque(maxlen=max_samples)
        self.rates: dict[str, deque[float]] = {n: deque(maxlen=max_samples) for n in self.names}
        net.add_post_step_hook(self._hook)

    def _hook(self, net: Network, idx: np.ndarray) -> None:
        if net.step_count % self.every:
            return
        counts = net.spike_count
        window = (net.t - self._t_last) / 1000.0
        if window <= 0:
            return
        delta = counts - self._last
        for n in self.names:
            s = net.populations[n].slice
            self.rates[n].append(float(delta[s].mean() / window))
        self.t.append(net.t)
        self._last = counts.copy()
        self._t_last = net.t

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return np.fromiter(self.t, float), np.fromiter(self.rates[name], float)


class Probe:
    """Samples arbitrary scalar functions every ``every_ms``."""

    def __init__(self, net: Network, signals: dict[str, Callable[[], float]],
                 every_ms: float = 10.0, max_samples: int = 100_000) -> None:
        self.signals = signals
        self.every = int(every_ms)
        self.t: deque[float] = deque(maxlen=max_samples)
        self.values: dict[str, deque[float]] = {k: deque(maxlen=max_samples) for k in signals}
        net.add_post_step_hook(self._hook)

    def _hook(self, net: Network, idx: np.ndarray) -> None:
        if net.step_count % self.every:
            return
        self.t.append(net.t)
        for k, fn in self.signals.items():
            self.values[k].append(float(fn()))

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return np.fromiter(self.t, float), np.fromiter(self.values[name], float)


def isi_cv(times_ms: np.ndarray, ids: np.ndarray) -> float:
    """Mean coefficient of variation of inter-spike intervals (1 ~ Poisson-like)."""
    cvs = []
    for nid in np.unique(ids):
        ts = np.sort(times_ms[ids == nid])
        if ts.size >= 4:
            isi = np.diff(ts)
            if isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else float("nan")


def power_spectrum(rate: np.ndarray, sample_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a population-rate signal (frequencies in Hz)."""
    x = rate - rate.mean()
    if x.size < 8:
        return np.zeros(0), np.zeros(0)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=sample_ms / 1000.0)
    return freqs, spec
