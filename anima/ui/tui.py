"""Live terminal dashboard (curses).

Shows the creature's world, drives, neuromodulators, regional activity, a
live spike raster of its cortex and the conversation, while you type.
Requires a terminal of at least 80x24.  On Windows install ``windows-curses``.
"""

from __future__ import annotations

import curses
import queue
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime.agent import Agent

SPARK = " ▁▂▃▄▅▆▇█"
REGIONS = [("cortex", "cortex.E"), ("senses", "sense.bearing"), ("hearing", "audio"),
           ("motor", "motor"), ("voice", "vocal.gate"), ("interoc.", "intero")]


def _bar(value: float, width: int, lo: float = 0.0, hi: float = 1.0) -> str:
    x = 0.0 if hi == lo else (value - lo) / (hi - lo)
    x = float(np.clip(x, 0.0, 1.0))
    full = int(round(x * width))
    return "█" * full + "·" * (width - full)


def _spark(values: list[float], width: int) -> str:
    """Sparkline scaled to the recent range, so fluctuations stay visible."""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 0.25 * max(abs(hi), 1.0))
    lo = max(0.0, hi - span)
    s = "".join(SPARK[int(np.clip((v - lo) / span * 8, 0, 8))] for v in vals)
    return s.rjust(width)


class Dashboard:
    def __init__(self, agent: Agent, brain_path: Path | None) -> None:
        self.agent = agent
        self.brain_path = brain_path
        self.events: queue.Queue = queue.Queue()
        self.chat: deque[tuple[str, str]] = deque(maxlen=200)
        self.rates: dict[str, deque[float]] = {k: deque(maxlen=120) for _, k in REGIONS}
        self.buffer = ""
        self.status = ""
        self.raster_cols = 60
        net = agent.brain.net
        cx = net.populations["cortex.E"]
        rng = np.random.default_rng(0)
        self.raster_ids = np.sort(rng.choice(np.arange(cx.start, cx.stop), size=min(10, cx.size),
                                             replace=False))
        self.raster = np.zeros((len(self.raster_ids), self.raster_cols), dtype=bool)
        self._last_counts = net.spike_count[self.raster_ids].copy()
        for kind in ("utterance", "heard", "stage", "ate"):
            agent.on(kind, lambda e: self.events.put(e))

    # ------------------------------------------------------------ helpers
    def _drain(self) -> None:
        while True:
            try:
                e = self.events.get_nowait()
            except queue.Empty:
                break
            t = f"{e.get('t', 0):7.1f}"
            k = e["kind"]
            if k == "utterance":
                self.chat.append(("anima", f"{t}  anima › {e['text']}"))
            elif k == "heard":
                self.chat.append(("you", f"{t}  you   › {e['text']}"))
            elif k == "stage":
                msg = {"nrem": "(falls asleep — slow-wave sleep, replaying the day)",
                       "rem": "(REM sleep — dreaming, body paralysed)",
                       "awake": "(wakes up)"}[e["stage"]]
                self.chat.append(("event", f"{t}  {msg}"))
            elif k == "ate":
                self.chat.append(("event", f"{t}  (eats)"))

    def _sample(self) -> None:
        snap = self.agent.latest
        for _, pop in REGIONS:
            if pop in snap.get("rates", {}):
                self.rates[pop].append(snap["rates"][pop])
        net = self.agent.brain.net
        counts = net.spike_count[self.raster_ids]
        fired = counts > self._last_counts
        self._last_counts = counts.copy()
        self.raster = np.roll(self.raster, -1, axis=1)
        self.raster[:, -1] = fired

    def _put(self, win: Any, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = win.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                win.addnstr(y, x, text, max(0, w - x - 1), attr)
            except curses.error:
                pass

    # ------------------------------------------------------------ drawing
    def draw(self, scr: Any) -> None:
        scr.erase()
        H, W = scr.getmaxyx()
        if H < 24 or W < 80:
            self._put(scr, 0, 0, f"window too small ({W}x{H}); need 80x24")
            scr.refresh()
            return
        s = self.agent.latest
        bold = curses.A_BOLD
        dim = curses.A_DIM
        c_title = curses.color_pair(1) | bold
        stage_col = {"awake": 2, "nrem": 3, "rem": 4}.get(s["stage"], 0)
        head = (f" anima  {s['clock']}  ")
        self._put(scr, 0, 0, head, c_title)
        self._put(scr, 0, len(head), f" {s['stage'].upper()} ", curses.color_pair(stage_col) | bold)
        info = (f"  age {s['sim_time_s']:7.1f}s  {s['speed']:.1f}x realtime  "
                f"{self.agent.brain.net.N} neurons  memories {s['memories']}  replays {s['replays']}")
        self._put(scr, 0, len(head) + 9, info, dim)
        # ---- world
        world_w = max(24, min(44, W // 2 - 4))
        world_h = 12
        rows = self.agent.world.render(world_w - 2, world_h - 2)
        self._put(scr, 2, 1, "world", bold)
        for i, r in enumerate(rows):
            self._put(scr, 3 + i, 1, r)
        if not rows:
            self._put(scr, 4, 2, "(empty world — pure inner life)", dim)
        doing = s.get("action") or "—"
        self._put(scr, 3 + max(len(rows), 2), 1, f"doing: {doing:8s} energy {_bar(s['energy'], 10)}")
        # ---- drives & modulators
        x0 = world_w + 4
        bw = max(10, W - x0 - 22)
        self._put(scr, 2, x0, "drives", bold)
        for i, (k, v) in enumerate(s["drives"].items()):
            self._put(scr, 3 + i, x0, f"{k:<11}{_bar(v, bw)} {v:4.2f}")
        self._put(scr, 8, x0, "neuromodulators", bold)
        m = s["modulators"]
        mods = [("dopamine", m["dopamine_tonic"], 0, 1), ("  phasic", m["dopamine"], -1, 1),
                ("noradren.", m["noradrenaline"], 0, 1), ("acetylch.", m["acetylcholine"], 0, 1),
                ("serotonin", m["serotonin"], 0, 1)]
        for i, (k, v, lo, hi) in enumerate(mods):
            self._put(scr, 9 + i, x0, f"{k:<11}{_bar(v, bw, lo, hi)} {v:+.2f}" if lo < 0
                      else f"{k:<11}{_bar(v, bw, lo, hi)} {v:4.2f}")
        # ---- activity
        y = 15
        self._put(scr, y, x0, "activity (Hz)", bold)
        sw = max(10, W - x0 - 20)
        for i, (label, pop) in enumerate(REGIONS):
            vals = list(self.rates[pop])
            cur = vals[-1] if vals else 0.0
            self._put(scr, y + 1 + i, x0, f"{label:<9}{_spark(vals, sw)} {cur:5.1f}")
        # ---- raster
        ry = 3 + world_h + 1
        self._put(scr, ry, 1, "cortex spikes (10 neurons, last 6 s)", bold)
        cols = min(self.raster_cols, world_w)
        for i in range(self.raster.shape[0]):
            line = "".join("|" if f else " " for f in self.raster[i, -cols:])
            self._put(scr, ry + 1 + i, 1, line, curses.color_pair(1))
        # ---- conversation
        cy = max(ry + 1 + self.raster.shape[0], y + 2 + len(REGIONS)) + 1
        self._put(scr, cy, 1, "conversation", bold)
        n = max(1, H - cy - 3)
        for i, (who, line) in enumerate(list(self.chat)[-n:]):
            attr = curses.color_pair(1) | bold if who == "anima" else (dim if who == "event" else 0)
            self._put(scr, cy + 1 + i, 1, line, attr)
        partial = s.get("speaking", "")
        if partial:
            self._put(scr, H - 2, 1, f"         anima is saying: {partial}…", curses.color_pair(1))
        if self.status:
            self._put(scr, H - 2, W - len(self.status) - 2, self.status, dim)
        self._put(scr, H - 1, 0, "you › " + self.buffer, bold)
        self._put(scr, H - 1, W - 41, "Enter: say  /feed  /save  /sleep  /wake  Esc: quit", dim)
        scr.move(H - 1, min(W - 2, 6 + len(self.buffer)))
        scr.refresh()

    # ------------------------------------------------------------ input
    def handle_line(self, line: str) -> bool:
        a = self.agent
        if not line.strip():
            return True
        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd in ("/quit", "/q", "/exit"):
                return False
            if cmd == "/save" and self.brain_path:
                a.save(self.brain_path)
                self.status = f"saved {self.brain_path}"
            elif cmd == "/sleep":
                a.call_soon(lambda ag: ag.brain.sleep.force("nrem", ag.brain.time_s))
            elif cmd == "/wake":
                a.call_soon(lambda ag: ag.brain.sleep.force("awake", ag.brain.time_s))
            elif cmd == "/feed":
                from .chat import feed
                parts = line.split()
                self.status = feed(a, parts[1] if len(parts) > 1 else "ahead")
            else:
                self.status = f"unknown {cmd}"
            return True
        a.hear(line)
        return True

    def loop(self, scr: Any, speed: float) -> None:
        curses.curs_set(1)
        scr.nodelay(True)
        scr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_CYAN, bg)
            curses.init_pair(2, curses.COLOR_GREEN, bg)
            curses.init_pair(3, curses.COLOR_BLUE, bg)
            curses.init_pair(4, curses.COLOR_MAGENTA, bg)
        self.agent.start(speed=speed)
        last_draw = 0.0
        try:
            while True:
                try:
                    ch = scr.get_wch()
                except curses.error:
                    ch = None
                if ch is not None:
                    if ch in ("\n", "\r", curses.KEY_ENTER):
                        line, self.buffer = self.buffer, ""
                        if not self.handle_line(line):
                            break
                    elif ch in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                        self.buffer = self.buffer[:-1]
                    elif ch == "\x1b":
                        break
                    elif isinstance(ch, str) and ch.isprintable():
                        self.buffer += ch
                now = time.time()
                if now - last_draw >= 0.1:
                    self._drain()
                    self._sample()
                    self.draw(scr)
                    last_draw = now
                time.sleep(0.01)
        finally:
            self.agent.stop()


def run_dashboard(agent: Agent, brain_path: Path | None, speed: float = 1.0, **_: Any) -> None:
    dash = Dashboard(agent, brain_path)
    try:
        curses.wrapper(lambda scr: dash.loop(scr, speed))
    except KeyboardInterrupt:
        agent.stop()
    if brain_path is not None:
        p = agent.save(brain_path)
        print(f"saved {p} — run again to continue its life.")
