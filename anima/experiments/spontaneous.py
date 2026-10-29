"""Experiment: a day in the life, with nobody there.

The agent is placed in an empty world and never receives any input.  We
record what its brain does on its own for one full simulated day:

* firing rates and spike irregularity (CV of inter-spike intervals);
* cortical rhythms awake vs. in slow-wave sleep (power spectrum of the
  population rate) — NREM should show slow (<2 Hz) up/down oscillations;
* the sleep/wake architecture produced by the two-process model;
* how much it moves and babbles without being prompted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.monitors import RateMonitor, isi_cv, power_spectrum
from ..runtime.agent import Agent


def run_day(seconds: float | None = None, seed: int = 1, day_length_s: float = 1200.0,
            verbose: bool = False) -> dict[str, Any]:
    agent = Agent.create(seed=seed, world="void", day_length_s=day_length_s)
    net = agent.brain.net
    seconds = day_length_s if seconds is None else seconds
    rm = RateMonitor(net, ["cortex.E", "cortex.I", "motor", "vocal.gate"], every_ms=10,
                     max_samples=int(seconds * 100) + 10)
    cx = net.populations["cortex.E"]
    ids = np.arange(cx.start, cx.start + 100)
    keep = np.zeros(net.N, dtype=bool)
    keep[ids] = True
    spikes_t: list[np.ndarray] = []
    spikes_i: list[np.ndarray] = []

    def hook(n, idx):
        if idx.size:
            k = idx[keep[idx]]
            if k.size:
                spikes_t.append(np.full(k.size, n.t))
                spikes_i.append(k)

    net.add_post_step_hook(hook)
    stage_t: list[float] = []
    stage_v: list[str] = []
    drives: list[dict[str, float]] = []
    utter: list[tuple[float, str]] = []
    actions = []
    agent.on("utterance", lambda e: utter.append((e["t"], e["text"])))
    agent.on("action", lambda e: actions.append(e["t"]))

    def snap(s: dict[str, Any]) -> None:
        stage_t.append(s["sim_time_s"])
        stage_v.append(s["stage"])
        drives.append(dict(s["drives"], **{f"mod_{k}": v for k, v in s["modulators"].items()}))

    agent.on("snapshot", snap)
    step = 60.0
    t = 0.0
    while t < seconds:
        agent.run(min(step, seconds - t))
        t += step
        if verbose:
            print(f"  t={t:6.0f}s  {agent.describe()}", flush=True)
    st = np.concatenate(spikes_t) if spikes_t else np.zeros(0)
    si = np.concatenate(spikes_i) if spikes_i else np.zeros(0, dtype=int)
    tr, rE = rm.series("cortex.E")
    stage_t_a = np.asarray(stage_t)
    stage_v_a = np.asarray(stage_v)
    # stage of every rate sample
    idx = np.clip(np.searchsorted(stage_t_a, tr / 1000.0), 0, len(stage_v_a) - 1)
    rate_stage = stage_v_a[idx] if len(stage_v_a) else np.array([])
    out: dict[str, Any] = {"seconds": seconds, "utterances": utter, "n_actions": len(actions),
                           "stage_t": stage_t, "stage": stage_v, "drives": drives,
                           "rate_t": tr.tolist(), "rate_cortex": rE.tolist(),
                           "spike_t": st.tolist(), "spike_i": si.tolist()}
    summary: dict[str, Any] = {}
    for stg in ("awake", "nrem", "rem"):
        sel = rate_stage == stg
        if sel.sum() > 300:
            seg = rE[sel]
            f, P = power_spectrum(seg[: (seg.size // 2) * 2], 10.0)
            band = (f > 0.2) & (f < 40)
            peak = float(f[band][np.argmax(P[band])]) if band.any() else float("nan")
            slow = float(P[(f > 0.2) & (f < 2)].sum() / max(P[(f > 0.2) & (f < 40)].sum(), 1e-12))
            # spike irregularity inside this stage
            ts_stage = idx_mask_times(st, stage_t_a, stage_v_a, stg)
            cv = isi_cv(st[ts_stage], si[ts_stage]) if ts_stage.any() else float("nan")
            summary[stg] = {"minutes": float(sel.sum() * 0.01 / 60.0), "cortex_rate_hz": float(seg.mean()),
                            "rate_std": float(seg.std()), "spectral_peak_hz": peak,
                            "slow_power_fraction": slow, "isi_cv": cv}
    summary["utterances"] = len(utter)
    summary["syllables"] = agent.body.stats["syllables"]
    summary["actions"] = len(actions)
    summary["fraction_asleep"] = float(np.mean(stage_v_a != "awake")) if len(stage_v_a) else 0.0
    summary["transitions"] = [(round(t, 1), s) for t, s in agent.brain.sleep.transitions]
    out["summary"] = summary
    return out


def idx_mask_times(times_ms: np.ndarray, stage_t: np.ndarray, stage_v: np.ndarray, stg: str) -> np.ndarray:
    if times_ms.size == 0 or stage_t.size == 0:
        return np.zeros(times_ms.size, dtype=bool)
    k = np.clip(np.searchsorted(stage_t, times_ms / 1000.0), 0, len(stage_v) - 1)
    return stage_v[k] == stg


def main(seconds: float | None = None, seed: int = 1, plot: str | None = None,
         verbose: bool = True) -> dict[str, Any]:
    res = run_day(seconds, seed=seed, verbose=verbose)
    if plot:
        _plot(res, plot)
    s = res["summary"]
    print("summary:")
    for k, v in s.items():
        if k != "transitions":
            print(f"  {k}: {v}")
    return s


def _plot(res: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = np.asarray(res["stage_t"])
    sv = np.asarray(res["stage"])
    level = np.select([sv == "awake", sv == "rem", sv == "nrem"], [2, 1, 0], 0)
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.step(st / 60.0, level, where="post", color="#456", lw=1.2)
    ax.set_yticks([0, 1, 2], ["NREM", "REM", "awake"])
    d = res["drives"]
    ax2 = ax.twinx()
    for key, color in (("hunger", "#c84"), ("fatigue", "#68c"), ("boredom", "#9a3"),
                       ("loneliness", "#c5a")):
        ax2.plot(st / 60.0, [x[key] for x in d], color=color, lw=1, label=key)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=4, frameon=False, fontsize=8)
    ax.set_title("one simulated day with no input: sleep stages and drives")
    ax.set_xlabel("minutes")
    rt = np.asarray(res["rate_t"]) / 1000.0
    rc = np.asarray(res["rate_cortex"])
    rate_stage = sv[np.clip(np.searchsorted(st, rt), 0, len(sv) - 1)]
    spk_t = np.asarray(res["spike_t"]) / 1000.0
    spk_i = np.asarray(res["spike_i"])
    for col, stg, title in ((0, "awake", "awake"), (1, "nrem", "slow-wave sleep")):
        idx = np.flatnonzero(rate_stage == stg)
        if idx.size < 600:
            continue
        mid = idx[idx.size // 2]
        t0 = rt[mid]
        w = (rt >= t0) & (rt < t0 + 6)
        a = fig.add_subplot(gs[1, col])
        sel = (spk_t >= t0) & (spk_t < t0 + 6)
        a.plot(spk_t[sel] - t0, spk_i[sel] - spk_i.min(), "|", ms=2, color="#234")
        a.set_title(f"cortex spikes, {title} (100 neurons, 6 s)")
        a.set_yticks([])
        a2 = fig.add_subplot(gs[2, col])
        a2.plot(rt[w] - t0, rc[w], color="#2a7", lw=0.8)
        a2.set(xlabel="seconds", ylabel="cortex rate (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
