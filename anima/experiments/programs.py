"""Experiments: does it learn which program to use, and is its thought log true?

``hands``  A creature with four hands lives in the foraging arena (a long
           day keeps it awake for the 20 minutes); each hand runs a program.  For the first two minutes none of them pays.
           Then the program on the hand it used *least* in those two minutes
           starts answering with a reward of +0.6, the other three with
           nothing.  Does it come to use the paying program more?  Control:
           the same creature with plasticity frozen at that moment
           (``frozen``) — both runs are identical until then.
``voice``  Two programs bound to what it says: syllables starting with "b"
           run one that pays +0.5 (after the first two minutes), syllables
           starting with "d" one that does not.  Does "b" become more common
           than "d"?  Control: plasticity frozen at minute 2.
``mind``   Is the thought log a faithful readout?  A creature is spoken to
           every ~12 s with a different syllable; we check (1) whether the
           intentions it logs come true — the next movement it commits to is
           the one it said it was inclined to make — compared with the base
           rate of that movement and with simply repeating the previous
           movement, and (2) whether the syllables it silently mouths while
           listening match what it heard, compared with chance.

Program answers carry no sound here (``speak = False``), so only their
consequences — reward or nothing — can shape behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mind import ThoughtStream
from ..runtime.agent import Agent
from ..tools import FunctionTool, ToolBelt

N_HANDS = 4


# ===================================================================== hands
def run_hands(seconds: float = 1200.0, seed: int = 1, plastic: bool = True, rewarded: int | None = None,
              baseline_s: float = 120.0, reward: float = 0.6, verbose: bool = False) -> dict[str, Any]:
    agent = Agent.create(seed=seed, hands=N_HANDS, day_length_s=7200.0)
    state: dict[str, Any] = {"paying": rewarded}

    def program(k: int):
        def fn(call: dict[str, Any]) -> dict[str, Any]:
            if k == state["paying"] and call["t"] >= baseline_s:
                return {"reward": reward}
            return {}
        return fn

    tools = [FunctionTool(name=f"program{k}", hand=k, inline=True, latency_s=0.3, speak=False, fn=program(k))
             for k in range(N_HANDS)]
    belt = ToolBelt(agent, tools)
    mind = ThoughtStream(agent, None, keep=100000, tool_names=belt.hand_names(), reflect_every_s=0)
    reaches: list[tuple[float, int]] = []
    actions: list[tuple[float, str]] = []
    readout: list[tuple[float, int, int, float]] = []  # at each reach: hand, the urge readout's leader, margin

    def on_hand(e: dict[str, Any]) -> None:
        reaches.append((e["t"], int(e["hand"])))
        lead, margin = mind._lead(mind._drive("hands"))
        readout.append((e["t"], int(e["hand"]), lead, round(margin, 4)))
    agent.on("hand", on_hand)
    agent.on("action", lambda e: actions.append((e["t"], e["action"])))
    step = 60.0
    t = 0.0
    while t < seconds:
        agent.run(min(step, seconds - t, baseline_s - t) if t < baseline_s else min(step, seconds - t))
        t = agent.t_ms / 1000.0
        if t >= baseline_s and "start" not in state:
            # the program on its least-used hand starts paying; the control freezes now
            counts = [sum(1 for _, h in reaches if h == k) for k in range(N_HANDS)]
            if state["paying"] is None:
                state["paying"] = int(np.argmin(counts))
            state["start"], state["baseline_counts"] = t, counts
            agent.brain.net.plasticity_enabled = plastic
        paying = state["paying"]
        if verbose:
            recent = [h for tt, h in reaches if tt >= t - 120]
            share = recent.count(paying) / len(recent) if recent and paying is not None else float("nan")
            print(f"  [hands{'' if plastic else '/frozen'} seed {seed}] t={t:5.0f}s reaches {len(reaches)} "
                  f"share of program{paying} (last 2 min) {share:.2f} energy {agent.body.energy:.2f}",
                  flush=True)
    belt.close()
    mind.close()
    return {"seed": seed, "plastic": plastic, "paying": state["paying"], "baseline_s": baseline_s,
            "baseline_counts": state.get("baseline_counts"), "seconds": seconds,
            "reaches": reaches, "readout": readout, "stats": belt.stats, "body": dict(agent.body.stats),
            "thoughts": mind.recent(0), "actions": actions}


def hand_curve(res: dict[str, Any], bin_s: float = 120.0) -> list[dict[str, float]]:
    out = []
    edges = list(np.arange(0.0, res["seconds"] - 1e-9, bin_s)) + [res["seconds"]]
    for a, b in zip(edges[:-1], edges[1:]):
        hs = [h for t, h in res["reaches"] if a <= t < b]
        counts = [hs.count(k) for k in range(N_HANDS)]
        out.append({"t0": float(a), "t1": float(b), "reaches_per_min": len(hs) / ((b - a) / 60.0),
                    "share": counts[res["paying"]] / len(hs) if hs else float("nan"), "counts": counts})
    return out


# ===================================================================== voice
def run_voice(seconds: float = 600.0, seed: int = 1, plastic: bool = True, good: str = "b", dud: str = "d",
              baseline_s: float = 120.0, reward: float = 0.5, verbose: bool = False) -> dict[str, Any]:
    agent = Agent.create(seed=seed, world="void")

    def pays(call: dict[str, Any]) -> dict[str, Any]:
        return {"reward": reward} if call["t"] >= baseline_s else {}

    belt = ToolBelt(agent, [FunctionTool(name="treat", say=good, inline=True, speak=False, cooldown_s=2.0, fn=pays),
                            FunctionTool(name="dud", say=dud, inline=True, speak=False, cooldown_s=2.0,
                                         fn=lambda c: {})])
    syl: list[tuple[float, str]] = []
    agent.on("syllable", lambda e: syl.append((e["t"], e["text"])))
    n_ticks = int(seconds * 1000 / agent.tick_ms)
    for k in range(n_ticks):
        agent.body.energy = 0.8  # well fed: we study the voice here (as in the babble experiment)
        if k == int(baseline_s * 1000 / agent.tick_ms):
            agent.brain.net.plasticity_enabled = plastic  # the control freezes when the program starts paying
        agent.tick()
        if verbose and k % 6000 == 0 and k:
            t = agent.t_ms / 1000.0
            recent = [s for tt, s in syl if tt >= t - 60]
            print(f"  [voice{'' if plastic else '/frozen'} seed {seed}] t={t:5.0f}s "
                  f"'{good}' {_share(recent, good):.2f} '{dud}' {_share(recent, dud):.2f} calls {belt.stats}",
                  flush=True)
    belt.close()
    return {"seed": seed, "plastic": plastic, "good": good, "dud": dud, "seconds": seconds,
            "baseline_s": baseline_s, "syllables": syl, "stats": belt.stats}


def _share(sylls: list[str], prefix: str) -> float:
    return sum(s.startswith(prefix) for s in sylls) / len(sylls) if sylls else float("nan")


def voice_curve(res: dict[str, Any], bin_s: float = 120.0) -> list[dict[str, float]]:
    out = []
    edges = list(np.arange(0.0, res["seconds"] - 1e-9, bin_s)) + [res["seconds"]]
    for a, b in zip(edges[:-1], edges[1:]):
        ss = [s for t, s in res["syllables"] if a <= t < b]
        out.append({"t0": float(a), "t1": float(b), "good": _share(ss, res["good"]),
                    "dud": _share(ss, res["dud"]), "syllables": len(ss)})
    return out


# ===================================================================== mind
TALK = ["ba ba", "do do", "mi mi", "ku ku", "se se", "lo lo", "pa pa", "ne ne", "ti ti", "wo wo", "gu gu", "he he"]


def run_mind(seconds: float = 480.0, seed: int = 1, train_s: float = 600.0, talk_every_s: float = 12.0,
             verbose: bool = False) -> dict[str, Any]:
    """``train_s`` of foraging first (mildly hungry, nobody around), then
    ``seconds`` in which someone says a syllable pair every ~12 s.  A long
    day keeps it awake throughout."""
    agent = Agent.create(seed=seed, day_length_s=7200.0)
    mind = ThoughtStream(agent, None, keep=100000)
    rng = np.random.default_rng(seed + 50)
    actions: list[tuple[float, str]] = []
    agent.on("action", lambda e: actions.append((e["t"], e["action"])))
    heard: list[tuple[float, str]] = []

    def cap(ag: Any) -> None:
        if ag.t_ms / 1000.0 < train_s and ag.body.energy > 0.5:
            ag.body.energy = 0.5  # mildly food-restricted while it learns to forage
    agent.add_tick_hook(cap)
    agent.run(train_s)
    next_talk = train_s + 5.0
    end = train_s + seconds
    while agent.t_ms / 1000.0 < end:
        t = agent.t_ms / 1000.0
        if t >= next_talk:
            phrase = TALK[int(rng.integers(len(TALK)))]
            agent.hear(phrase)
            heard.append((t, phrase))
            next_talk = t + talk_every_s * (0.75 + 0.5 * rng.random())
        agent.run(min(0.5, end - t))
    mind.close()
    if verbose:
        print(f"  [mind seed {seed}] {mind.stats()}", flush=True)
    return {"seed": seed, "seconds": seconds, "train_s": train_s, "thoughts": mind.recent(0), "actions": actions,
            "heard": heard, "unconscious": mind.unconscious, "ate": agent.body.stats.get("ate", 0)}


def _majority(at: np.ndarray, al: np.ndarray, t0: float, t1: float, n: int) -> int:
    sel = al[(at > t0) & (at <= t1)]
    return int(np.bincount(sel, minlength=n).argmax()) if len(sel) else -1


def intention_check(thoughts: list[dict[str, Any]], actions: list[tuple[float, str]],
                    window_s: float = 3.0) -> dict[str, Any]:
    """Does a logged inclination ("I'm inclined to turn left") come true?

    Target: the movement it makes most in the next ``window_s`` seconds.
    Baselines: ``chance`` — how often that movement is the most common one
    in any 3 s of the run; ``persistence`` — guessing that the next 3 s go
    like the last 3 s; ``already_doing`` — how often the inclination merely
    names what it was already doing most."""
    acts = sorted(actions)
    if len(acts) < 10:
        return {"n": 0}
    labels = sorted({lab for _, lab in acts})
    at = np.array([t for t, _ in acts])
    al = np.array([labels.index(lab) for _, lab in acts])
    k = len(labels)
    grid = np.arange(at[0], at[-1] - window_s, 0.5)
    majs = [m for m in (_majority(at, al, g, g + window_s, k) for g in grid) if m >= 0]
    p = np.bincount(majs, minlength=k) / max(1, len(majs))
    n = hit = already = persist = 0
    chance = 0.0
    for th in thoughts:
        move = th.get("data", {}).get("move")
        if th["kind"] != "intention" or move not in labels:
            continue
        t, mv = th["t"], labels.index(move)
        nxt = _majority(at, al, t, t + window_s, k)
        prev = _majority(at, al, t - window_s, t, k)
        if nxt < 0:
            continue
        n += 1
        hit += nxt == mv
        chance += p[mv]
        already += prev == mv
        persist += prev == nxt
    if not n:
        return {"n": 0}
    return {"n": n, "came_true": hit / n, "chance": chance / n, "persistence": persist / n,
            "already_doing": already / n, "window_s": window_s}


def urge_check(thoughts: list[dict[str, Any]], reaches: list[tuple[float, int]], paying: int | None,
               baseline_s: float, horizon_s: float = 30.0,
               readout: list[tuple[float, int, int, float]] | None = None) -> dict[str, Any]:
    """Does "I feel like using X (hand k)" come true, and does it come first?

    Logged urges: is the next reach (within ``horizon_s``) the hand named?
    Baselines: the hand of the previous reach, and the most used hand among
    the last six reaches.  The readout behind them, at every reach: is the
    hand it favours the one that reaches (against the previous reach's
    hand)?  Timing: when the urge first named the paying program, and when
    the paying program first made up half of the last six reaches."""
    rs = sorted(reaches)
    rt = np.array([t for t, _ in rs])
    rh = np.array([h for _, h in rs])
    n = hit = last_hit = recent_hit = 0
    for th in thoughts:
        hand = th.get("data", {}).get("hand")
        if th["kind"] != "intention" or hand is None:
            continue
        t = th["t"]
        i = int(np.searchsorted(rt, t, side="right"))
        if i >= len(rt) or rt[i] - t > horizon_s or i == 0:
            continue
        n += 1
        hit += rh[i] == hand
        last_hit += rh[i] == rh[i - 1]
        recent = rh[max(0, i - 6):i]
        recent_hit += rh[i] == np.bincount(recent, minlength=4).argmax()
    out: dict[str, Any] = {"n": n}
    if n:
        out.update({"came_true": hit / n, "previous_reach": last_hit / n, "recent_majority": recent_hit / n,
                    "horizon_s": horizon_s})
    if readout:
        ro = [r for r in readout if r[0] >= baseline_s]
        if len(ro) > 1:
            out["readout_at_reach"] = {
                "reaches": len(ro), "agrees": float(np.mean([h == lead for _, h, lead, _ in ro])),
                "previous_reach": float(np.mean([ro[i][1] == ro[i - 1][1] for i in range(1, len(ro))])),
                "agrees_when_clear": (float(np.mean([h == lead for _, h, lead, m in ro if m >= 0.06]))
                                      if any(m >= 0.06 for *_, m in ro) else None),
                "clear_share": float(np.mean([m >= 0.06 for *_, m in ro]))}
    if paying is not None:
        first_urge = next((th["t"] for th in thoughts if th["kind"] == "intention" and th["t"] >= baseline_s
                           and th.get("data", {}).get("hand") == paying), None)
        first_habit = None
        for i in range(len(rt)):
            if rt[i] < baseline_s:
                continue
            last6 = rh[max(0, i - 5):i + 1]
            if len(last6) == 6 and (last6 == paying).sum() >= 3:
                first_habit = float(rt[i])
                break
        out.update({"first_urge_s": first_urge, "first_habit_s": first_habit})
    return out


def echo_check(thoughts: list[dict[str, Any]], consonants: str = "bdgklmnpstwh") -> dict[str, Any]:
    """Do the syllables it mouths while listening start with the consonant it heard?"""
    trials = []
    for th in thoughts:
        if th["kind"] != "inner":
            continue
        heard = th["data"].get("heard", "")
        c = next((ch for ch in heard if ch in consonants), None)
        syl = th["data"].get("syllables", [])
        if c is None or not syl:
            continue
        trials.append((c, syl))
    if not trials:
        return {"n": 0}
    first = [s[0][0] for _, s in trials]
    match_first = np.mean([c == f for (c, _), f in zip(trials, first)])
    match_any = np.mean([any(x.startswith(c) for x in s) for c, s in trials])
    # chance: how often the first mouthed syllable starts with c on trials where something else was heard
    chance_first, chance_any = [], []
    for i, (c, s) in enumerate(trials):
        others = [f for j, ((c2, _), f) in enumerate(zip(trials, first)) if c2 != c]
        if others:
            p = others.count(c) / len(others)
            chance_first.append(p)
            chance_any.append(1.0 - (1.0 - p) ** len(s))
    def mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else float("nan")
    return {"n": len(trials), "first_matches": float(match_first), "first_chance": mean(chance_first),
            "any_matches": float(match_any), "any_chance": mean(chance_any),
            "syllables_per_trial": float(np.mean([len(s) for _, s in trials]))}


def thought_rates(thoughts: list[dict[str, Any]], seconds: float) -> dict[str, float]:
    kinds: dict[str, int] = {}
    for th in thoughts:
        kinds[th["kind"]] = kinds.get(th["kind"], 0) + 1
    return {"per_min": len(thoughts) / (seconds / 60.0), **{k: v for k, v in sorted(kinds.items())}}


# ===================================================================== main
def _job(args: tuple[str, int, bool, float]) -> tuple[str, int, bool, dict[str, Any]]:
    kind, seed, plastic, seconds = args
    if kind == "hands":
        return kind, seed, plastic, run_hands(seconds, seed=seed, plastic=plastic)
    if kind == "voice":
        return kind, seed, plastic, run_voice(seconds, seed=seed, plastic=plastic)
    return kind, seed, plastic, run_mind(seconds, seed=seed)


def summarize(raw: list[tuple[str, int, bool, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"hands": {"plastic": [], "frozen": []}, "voice": {"plastic": [], "frozen": []},
                           "mind": []}
    for kind, seed, plastic, res in sorted(raw, key=lambda r: (r[0], r[1], not r[2])):
        cond = "plastic" if plastic else "frozen"
        if kind == "hands":
            curve = hand_curve(res)
            after = [h for t, h in res["reaches"] if t >= res["seconds"] - 600]
            out["hands"][cond].append({
                "seed": seed, "paying": res["paying"], "baseline_counts": res["baseline_counts"],
                "curve": curve, "stats": res["stats"],
                "share_baseline": curve[0]["share"],
                "share_last_10min": after.count(res["paying"]) / len(after) if after else float("nan"),
                "food_eaten": res["body"].get("ate", 0),
                "intentions": intention_check(res["thoughts"], res["actions"]),
                "urges": urge_check(res["thoughts"], res["reaches"], res["paying"], res["baseline_s"],
                                    readout=res.get("readout")),
                "thought_rates": thought_rates(res["thoughts"], res["seconds"])})
        elif kind == "voice":
            curve = voice_curve(res)
            last = [s for t, s in res["syllables"] if t >= res["seconds"] - 240]
            out["voice"][cond].append({
                "seed": seed, "curve": curve, "stats": res["stats"],
                "good_baseline": curve[0]["good"], "dud_baseline": curve[0]["dud"],
                "good_last_4min": _share(last, res["good"]), "dud_last_4min": _share(last, res["dud"])})
        else:
            tr = res["train_s"]
            first = [x for x in res["thoughts"] if x["t"] < min(300.0, tr)]
            later = [x for x in res["thoughts"] if x["t"] >= tr]
            out["mind"].append({
                "seed": seed, "train_s": tr, "food_eaten": res["ate"],
                "intentions_first_5min": intention_check(first, [a for a in res["actions"] if a[0] < 300.0]),
                "intentions": intention_check(later, [a for a in res["actions"] if a[0] >= tr - 3.0]),
                "echo": echo_check(later),
                "thought_rates": thought_rates(later, res["seconds"]),
                "unconscious": res["unconscious"],
                "sample": [f"{t['clock']}  {t['kind']:<10} {t['text']}" for t in later[:80]]})
    return out


def main(seconds: float | None = None, seeds: tuple[int, ...] = (1, 2), parts: tuple[str, ...] = ("hands", "voice", "mind"),
         workers: int = 1, plot: str | None = None, verbose: bool = True) -> dict[str, Any]:
    jobs = []
    for s in seeds:
        if "hands" in parts:
            jobs += [("hands", s, True, seconds or 1200.0), ("hands", s, False, seconds or 1200.0)]
        if "voice" in parts:
            jobs += [("voice", s, True, seconds or 600.0), ("voice", s, False, seconds or 600.0)]
        if "mind" in parts:
            jobs.append(("mind", s, True, seconds or 480.0))  # after 600 s of foraging
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            raw = list(ex.map(_job, jobs))
    else:
        raw = [_job(j) for j in jobs]
    res = summarize(raw)
    if verbose:
        print(report(res))
    if plot:
        _plot(res, plot)
    return res


def report(res: dict[str, Any]) -> str:
    lines = []
    for cond in ("plastic", "frozen"):
        runs = res["hands"][cond]
        if runs:
            lines.append(f"hands/{cond}: share of reaches with the paying program, first 2 min -> last 10 min: "
                         + "; ".join(f"seed {r['seed']} (program{r['paying']}) {r['share_baseline']:.2f} -> "
                                     f"{r['share_last_10min']:.2f}" for r in runs) + "  (chance 0.25)")
    for cond in ("plastic", "frozen"):
        runs = res["voice"][cond]
        if runs:
            lines.append(f"voice/{cond}: share of 'b' vs 'd' syllables, first 2 min -> last 4 min: "
                         + "; ".join(f"seed {r['seed']} b {r['good_baseline']:.3f} -> {r['good_last_4min']:.3f}, "
                                     f"d {r['dud_baseline']:.3f} -> {r['dud_last_4min']:.3f}" for r in runs))
    for cond in ("plastic", "frozen"):
        for r in res["hands"][cond]:
            u = r["urges"]
            if u.get("n"):
                lines.append(f"hands/{cond} seed {r['seed']}: {u['n']} urges, the next reach used that hand "
                             f"{u['came_true']:.2f} (previous reach's hand {u['previous_reach']:.2f}, most used of "
                             f"the last six {u['recent_majority']:.2f}); first urge for the paying program at "
                             f"{u.get('first_urge_s')} s, habit (3 of 6 reaches) at {u.get('first_habit_s')} s")
            ro = u.get("readout_at_reach")
            if ro:
                lines.append(f"hands/{cond} seed {r['seed']}: at {ro['reaches']} reaches the urge readout favoured "
                             f"the hand that reached {ro['agrees']:.2f} of the time (previous reach's hand "
                             f"{ro['previous_reach']:.2f}); clear ({ro['clear_share']:.2f} of reaches): "
                             f"{ro['agrees_when_clear']}")
    for r in res["mind"] + [x for c in ("plastic", "frozen") for x in res["hands"][c]]:
        i = r["intentions"]
        if i.get("n"):
            lines.append(f"seed {r['seed']} ({'mind' if 'echo' in r else 'hands'}): {i['n']} inclinations, the "
                         f"next {i['window_s']:g} s went that way {i['came_true']:.2f} (chance {i['chance']:.2f}, "
                         f"same as the last {i['window_s']:g} s {i['persistence']:.2f}; named what it was already "
                         f"doing {i['already_doing']:.2f})")
    for r in res["mind"]:
        i = r["intentions_first_5min"]
        if i.get("n"):
            lines.append(f"seed {r['seed']} (mind, first 5 min of its life): {i['n']} inclinations came true "
                         f"{i['came_true']:.2f} (chance {i['chance']:.2f}, persistence {i['persistence']:.2f})")
        e = r["echo"]
        if e.get("n"):
            lines.append(f"mind seed {r['seed']}: {e['n']} silent echoes, first syllable matches what it heard "
                         f"{e['first_matches']:.2f} (chance {e['first_chance']:.2f}); any syllable "
                         f"{e['any_matches']:.2f} (chance {e['any_chance']:.2f})")
        lines.append(f"mind seed {r['seed']}: {r['thought_rates']['per_min']:.1f} thoughts per minute")
    return "\n".join(lines)


def _plot(res: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for cond, color in (("plastic", "#2a7"), ("frozen", "#999")):
        runs = res["hands"][cond]
        if not runs:
            continue
        x = [(c["t0"] + c["t1"]) / 120.0 for c in runs[0]["curve"]]
        ys = np.array([[c["share"] for c in r["curve"]] for r in runs], dtype=float)
        ax.plot(x, np.nanmean(ys, axis=0), "-o", color=color, label=cond)
        for y in ys:
            ax.plot(x, y, color=color, alpha=0.25, lw=1)
    ax.axhline(0.25, color="k", lw=0.6, ls="--")
    ax.text(0.98, 0.26, "chance", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.axvline(2.0, color="k", lw=0.6, ls=":")
    ax.set(title="reaches that run the paying program", xlabel="minutes (it pays from 2)", ylim=(0, 1.02))
    ax.set_xticks(range(0, 21, 4))
    ax.legend(frameon=False)
    ax = axes[1]
    for cond, color in (("plastic", "#2a7"), ("frozen", "#999")):
        runs = res["voice"][cond]
        if not runs:
            continue
        x = [(c["t0"] + c["t1"]) / 120.0 for c in runs[0]["curve"]]
        good = np.array([[c["good"] for c in r["curve"]] for r in runs], dtype=float)
        dud = np.array([[c["dud"] for c in r["curve"]] for r in runs], dtype=float)
        ax.plot(x, np.nanmean(good, axis=0), "-o", color=color, label=f"'b' (pays), {cond}")
        ax.plot(x, np.nanmean(dud, axis=0), "--", color=color, label=f"'d' (does not), {cond}")
    ax.axvline(2.0, color="k", lw=0.6, ls=":")
    ax.set(title="syllables that run a program", xlabel="minutes (it pays from 2)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
