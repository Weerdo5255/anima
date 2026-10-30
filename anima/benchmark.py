"""Can anima run on this computer's GPU, and how fast?  (``anima gpu-check``,
``anima bench``.)

:func:`gpu_check` runs the device engine next to the reference CPU engine
on the same small brain and reports

1. whether their spike trains agree exactly with plasticity off (the same
   noise, the device in float64 if it has it — Apple's ``mps`` does not, and
   then the check is for close agreement);
2. how closely the learned weights and eligibility traces agree with
   plasticity on, before the two runs drift apart;
3. how fast a whole creature lives on each.

:func:`bench` measures speed (simulated seconds per wall-clock second) by
brain size.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .brain.builder import build_network
from .brain.presets import infant_spec
from .core.device import DeviceSpec, describe_device, resolve_device


def _pair(device: DeviceSpec, cortex: int, seed: int, dtype: Any) -> tuple[Any, Any]:
    spec = infant_spec(cortex_size=cortex, seed=seed)
    ref = build_network(spec)
    dev = build_network(spec, device=device,
                        engine_options={"dtype": dtype, "noise": "host", "propagation": "sparse"})
    for net in (ref, dev):
        net.I_ext[:] = 2.0
        net.params_changed()
    return ref, dev


def agreement(device: Any = "emulate", steps: int = 2000, cortex: int = 200, seed: int = 3) -> dict[str, Any]:
    """Spike trains of the reference and the device engine, plasticity off."""
    spec = resolve_device(device)
    dtype = np.float32 if spec.device == "mps" else np.float64
    ref, dev = _pair(spec, cortex, seed, dtype)
    for net in (ref, dev):
        net.plasticity_enabled = False
    first = None
    for k in range(steps):
        ref.step()
        dev.step()
        if first is None and k % 10 == 9 and not np.array_equal(ref.spike_count, dev.spike_count):
            first = k + 1
    dv = float(np.max(np.abs(ref.v - dev.v)))
    return {"steps": steps, "dtype": np.dtype(dtype).name, "identical": first is None, "first_difference_ms": first,
            "spikes_cpu": int(ref.spike_count.sum()), "spikes_device": int(dev.spike_count.sum()),
            "max_dv_mV": dv}


def plasticity_agreement(device: Any = "emulate", steps: int = 150, cortex: int = 200,
                         seed: int = 3) -> dict[str, Any]:
    """Learned weights (and eligibility traces) of both engines after ``steps``
    ms with plasticity on, as the largest difference relative to the
    largest value, per learning rule."""
    spec = resolve_device(device)
    dtype = np.float32 if spec.device == "mps" else np.float64
    ref, dev = _pair(spec, cortex, seed, dtype)
    for k in range(steps):
        for net in (ref, dev):
            net.modulators["dopamine"] = 0.4 if (k // 50) % 2 == 0 else -0.3
        ref.step()
        dev.step()
    out: dict[str, float] = {}
    for name, p in ref.projections.items():
        if p.rule is None:
            continue
        a, b = ref.weights(name), dev.weights(name)
        d = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-12))
        kind = type(p.rule).__name__
        if kind == "RewardSTDP":
            ea, eb = ref.eligibility(name), dev.eligibility(name)
            d = max(d, float(np.max(np.abs(ea - eb)) / max(float(np.max(np.abs(ea))), 1e-12)))
        out[kind] = max(out.get(kind, 0.0), d)
    return {"steps": steps, "same_spikes": bool(np.array_equal(ref.spike_count, dev.spike_count)),
            "max_relative_difference": out}


def speed(device: Any, cortex: int = 1000, seconds: float = 3.0, warmup: float = 0.5, seed: int = 7) -> dict[str, Any]:
    """A whole creature (brain, body, world) living as fast as it can."""
    from .runtime.agent import Agent
    t_build = time.perf_counter()
    agent = Agent.create(seed=seed, device=device, cortex_size=cortex)
    t_build = time.perf_counter() - t_build
    agent.run(seconds=warmup)
    agent.brain.net.sync()
    t0 = time.perf_counter()
    agent.run(seconds=seconds)
    agent.brain.net.sync()
    wall = time.perf_counter() - t0
    return {"device": agent.device, "cortex": cortex, "neurons": int(agent.brain.net.N),
            "x_real_time": seconds / wall, "ms_per_sim_second": 1000.0 * wall / seconds, "build_s": t_build}


def gpu_check(device: Any = "auto", say: Callable[[str], None] = print, quick: bool = False) -> dict[str, Any]:
    """The checks of ``anima gpu-check``; returns their numbers."""
    from .core.device import available_gpus
    spec = resolve_device(device)
    say(f"device: {describe_device(spec)}")
    if spec.is_reference:
        found = available_gpus()
        say("no usable GPU found" + ("" if not found else f" (found: {', '.join(map(str, found))})")
            + " — install CuPy or PyTorch, or pass --device emulate to check the device engine on the CPU")
        return {"device": "cpu", "ok": False}
    res: dict[str, Any] = {"device": str(spec)}
    a = agreement(spec, steps=500 if quick else 2000)
    res["agreement"] = a
    if a["identical"]:
        say(f"1. spike trains, plasticity off: identical to the CPU engine for {a['steps']} ms ({a['dtype']})")
    else:
        say(f"1. spike trains, plasticity off ({a['dtype']}): first difference after {a['first_difference_ms']} ms; "
            f"{a['spikes_device']} spikes vs {a['spikes_cpu']} on the CPU (float32 differs from float64 "
            f"in the last digits, and spiking networks amplify such differences — the counts should be close)")
    pa = plasticity_agreement(spec, steps=150)
    res["plasticity"] = pa
    rel = ", ".join(f"{k} {v:.1e}" for k, v in sorted(pa["max_relative_difference"].items()))
    say(f"2. learning, first 150 ms: largest relative weight/eligibility difference — {rel} "
        f"(STDP is applied per 10 ms window on a device, so ~1e-4 there is expected)")
    sizes = [1000] if quick else [1000, 4000]
    res["speed"] = []
    for cortex in sizes:
        d = speed(spec, cortex=cortex, seconds=1.0 if quick else 3.0)
        c = speed("cpu", cortex=cortex, seconds=1.0 if quick else 3.0)
        res["speed"].append({"device": d, "cpu": c})
        say(f"3. speed, cortex {cortex} ({d['neurons']} neurons): {d['x_real_time']:.2f}x real time on "
            f"{d['device']}, {c['x_real_time']:.2f}x on the CPU")
    worst = max(pa["max_relative_difference"].values()) if pa["max_relative_difference"] else 0.0
    res["ok"] = bool((a["identical"] or abs(a["spikes_device"] - a["spikes_cpu"]) <= 0.1 * max(a["spikes_cpu"], 1))
                     and worst < 0.05)
    say("result: " + ("the device engine agrees with the CPU engine" if res["ok"]
                      else "the device engine DISAGREES with the CPU engine — please report this"))
    return res


def bench(device: Any = "auto", cortex_sizes: tuple[int, ...] = (1000, 4000, 10000), seconds: float = 3.0,
          cpu: bool = True, say: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Speed by brain size on ``device`` (and the CPU, unless ``cpu`` is False)."""
    spec = resolve_device(device)
    say(f"device: {describe_device(spec)}")
    rows = []
    devices = [spec] + ([resolve_device("cpu")] if cpu and not spec.is_reference else [])
    for cortex in cortex_sizes:
        for dev in devices:
            r = speed(dev, cortex=cortex, seconds=seconds)
            rows.append(r)
            say(f"  cortex {cortex:>6} ({r['neurons']:>6} neurons) on {r['device']:<14} "
                f"{r['x_real_time']:6.2f}x real time  ({r['ms_per_sim_second']:7.0f} ms per simulated second, "
                f"built in {r['build_s']:.1f} s)")
    return rows


__all__ = ["gpu_check", "bench", "agreement", "plasticity_agreement", "speed"]
