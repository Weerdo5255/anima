"""A plain terminal conversation that works everywhere.

The creature lives in a background thread; this module only prints what it
does and forwards what you type.  It never waits for you and you never wait
for it: it may babble while you are typing, and you can speak while it is.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..runtime.agent import Agent

try:  # readline lets us redraw the half-typed input line after printing
    import readline  # noqa: F401
    _HAVE_READLINE = True
except Exception:  # pragma: no cover - Windows
    _HAVE_READLINE = False

HELP = """commands:
  /status        drives, neuromodulators, sleep stage
  /world         show the foraging arena
  /stats         lifetime counters
  /feed [where]  give it food: ahead (default), left, right or near
  /save [file]   save the creature
  /sleep /wake   force a sleep stage
  /quiet         toggle printing of non-speech events
  /help          this text
  /quit          save and leave
anything else is spoken to it.  :) or *smile* is approval, :( disapproval."""


def feed(agent: Agent, where: str = "ahead") -> str:
    """The ``/feed`` command: put a piece of food near the creature."""
    from ..world.foraging import OFFER_PLACES
    where = where.lower()
    if not hasattr(agent.world, "offer_food"):
        return "there is no food in this world"
    if where not in OFFER_PLACES:
        return f"/feed {' | '.join(OFFER_PLACES)}"
    agent.call_soon(lambda ag: ag.world.offer_food(where))
    return f"(you put some food {where if where != 'near' else 'nearby'})"


class ChatUI:
    def __init__(self, agent: Agent, brain_path: Path | None, show_events: bool = True,
                 out: Callable[[str], None] | None = None) -> None:
        self.agent = agent
        self.brain_path = brain_path
        self.show_events = show_events
        self._q: queue.Queue = queue.Queue()
        self._print_lock = threading.Lock()
        self._out = out or self._default_out
        agent.on("utterance", lambda e: self._q.put(("utterance", e)))
        agent.on("stage", lambda e: self._q.put(("stage", e)))
        agent.on("ate", lambda e: self._q.put(("ate", e)))
        self._stop = threading.Event()

    # ------------------------------------------------------------ output
    def _default_out(self, line: str) -> None:
        with self._print_lock:
            if _HAVE_READLINE and sys.stdin.isatty():
                import readline
                buf = readline.get_line_buffer()
                sys.stdout.write("\r\033[K" + line + "\n" + "you › " + buf)
            else:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _printer(self) -> None:
        while not self._stop.is_set():
            try:
                kind, e = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            snap = self.agent.latest
            stamp = f"{snap.get('clock', '--:--')}"
            if kind == "utterance":
                self._out(f"\033[1;36manima ›\033[0m {e['text']}   \033[2m[{stamp}]\033[0m")
            elif kind == "stage" and self.show_events:
                msg = {"nrem": "falls asleep", "rem": "is dreaming (REM)",
                       "awake": "wakes up"}.get(e["stage"], e["stage"])
                if e["stage"] == "nrem" and e.get("from") == "rem":
                    msg = "sinks back into deep sleep"
                self._out(f"\033[2m      (anima {msg} at {stamp})\033[0m")
            elif kind == "ate" and self.show_events:
                self._out("\033[2m      (anima found food)\033[0m")

    # ------------------------------------------------------------ commands
    def command(self, line: str) -> bool:
        """Handle a slash command; return False to quit."""
        parts = line.strip().split()
        cmd = parts[0].lower()
        a = self.agent
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            self._out(HELP)
        elif cmd == "/status":
            self._out(a.describe())
        elif cmd == "/stats":
            s = a.latest
            self._out(f"age {s['sim_time_s']:.0f}s  {s['stats']}  memories {s['memories']}  "
                      f"replays {s['replays']}  speed {s['speed']:.2f}x")
        elif cmd == "/world":
            for row in a.world.render(40, 16):
                self._out("  " + row)
        elif cmd == "/save":
            path = Path(parts[1]) if len(parts) > 1 else self.brain_path
            if path is None:
                self._out("no file given")
            else:
                p = a.save(path)
                self._out(f"saved to {p}")
        elif cmd == "/feed":
            self._out(feed(a, parts[1] if len(parts) > 1 else "ahead"))
        elif cmd == "/sleep":
            a.call_soon(lambda ag: ag.brain.sleep.force("nrem", ag.brain.time_s))
        elif cmd == "/wake":
            a.call_soon(lambda ag: ag.brain.sleep.force("awake", ag.brain.time_s))
        elif cmd == "/quiet":
            self.show_events = not self.show_events
            self._out(f"events {'shown' if self.show_events else 'hidden'}")
        else:
            self._out(f"unknown command {cmd}; /help for help")
        return True

    # ------------------------------------------------------------ main loop
    def run(self, speed: float = 1.0, autosave_s: float = 300.0) -> None:
        import os
        if os.name == "nt":  # enable ANSI colours in Windows consoles
            os.system("")
        a = self.agent
        printer = threading.Thread(target=self._printer, daemon=True)
        printer.start()
        a.start(speed=speed)
        self._out(f"\033[1manima\033[0m is alive ({a.brain.net.N} spiking neurons, "
                  f"{a.latest['clock']} {a.latest['stage']}). It does not wait for you: "
                  f"it will babble, wander, get hungry, bored, lonely and sleepy on its own.")
        self._out("Talk to it whenever you like. /help for commands, /quit to leave.\n")
        last_save = time.time()
        try:
            while True:
                try:
                    line = input("you › ")
                except EOFError:
                    break
                if autosave_s and self.brain_path and time.time() - last_save > autosave_s:
                    a.save(self.brain_path)
                    last_save = time.time()
                if not line.strip():
                    continue
                if line.startswith("/"):
                    if not self.command(line):
                        break
                    continue
                a.hear(line)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            a.stop()
            if self.brain_path is not None:
                p = a.save(self.brain_path)
                print(f"\nsaved {p} — run again to continue its life.")


def run_chat(agent: Agent, brain_path: Path | None, speed: float = 1.0,
             show_events: bool = True, **_: Any) -> None:
    ChatUI(agent, brain_path, show_events).run(speed=speed)
