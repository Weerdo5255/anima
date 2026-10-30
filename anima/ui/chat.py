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
  /pleasure [0-1] [for]   give it pleasure for what it just did (default 0.6, for its
                          answer to what you last said, or else its last words or
                          movement; for = speech, move or now).  Shortcut: +  ++  +++
  /pain [0-1] [for]       give it pain for what it just did.  Shortcut: -  --  ---
  /status        drives, feelings, neuromodulators, sleep stage
  /world         show the foraging arena
  /stats         lifetime counters
  /feed [where]  give it food: ahead (default), left, right or near
  /thoughts [n]  its latest thoughts (read out of its brain activity)
  /mind          toggle printing its thoughts as they happen
  /tools         the programs it may use, and how they have gone
  /save [file]   save the creature
  /sleep /wake   force a sleep stage
  /quiet         toggle printing of non-speech events
  /help          this text
  /quit          save and leave
anything else is spoken to it.  :) or *smile* is approval, :( disapproval.
Pleasure and pain teach far more than a smile or a frown, and reach the right thing even when
you are a few seconds late: it recalls what it just said (or did) while it feels them."""

#: what + ++ +++ and - -- --- give
SHORTCUTS = {"+": ("pleasure", 0.5), "++": ("pleasure", 0.8), "+++": ("pleasure", 1.0),
             "-": ("pain", 0.4), "--": ("pain", 0.7), "---": ("pain", 1.0)}
_FOR = {"speech": "speech", "words": "speech", "say": "speech", "said": "speech", "move": "movement",
        "movement": "movement", "moves": "movement", "now": "now", "auto": "auto"}


class ReplyTracker:
    """Follows its answer to what you last said, so that ``+`` / ``-`` (and
    ``/pleasure`` / ``/pain`` without a "for") reach exactly that answer —
    even if it has babbled on since — when it came in the last ``recent_s``
    seconds; otherwise they are for its last words or movement ("auto")."""

    def __init__(self, agent: Agent, recent_s: float = 15.0) -> None:
        from ..tutor.skills import WordTrial
        self._Trial = WordTrial
        self.agent = agent
        self.recent_s = recent_s
        self._trial: Any = None
        self.last: dict[str, Any] | None = None  # the last complete answer: acts, said, reply, t
        self._was = False
        agent.on("syllable", self._on_syllable)
        agent.add_tick_hook(self._on_tick)

    def heard(self, text: str) -> None:
        """You said ``text`` (call right after ``agent.hear``)."""
        self.agent.call_soon(lambda ag: self._start(text))

    def _start(self, text: str) -> None:
        t = self._Trial(text, self.agent.t_ms / 1000.0)
        self._trial = t if t.target else None

    def _on_syllable(self, e: dict[str, Any]) -> None:
        t = self._trial
        if t is not None and t.syllable(e["t"], e["text"], e.get("act")):
            self._done(t)

    def _on_tick(self, agent: Agent) -> None:
        ear = agent.body.ear
        hearing = bool(ear is not None and ear.external_pending)
        t = self._trial
        if t is not None:
            if self._was and not hearing:
                t.heard_end(agent.t_ms / 1000.0)
            if t.tick(agent.t_ms / 1000.0):
                self._done(t)
        self._was = hearing

    def _done(self, t: Any) -> None:
        self._trial = None
        if t.acts:
            self.last = {"acts": list(t.acts), "said": " ".join(t.target), "reply": " ".join(t.reply),
                         "t": self.agent.t_ms / 1000.0}

    def credit(self) -> Any:
        last = self.last
        if last is not None and self.agent.t_ms / 1000.0 - last["t"] <= self.recent_s:
            return list(last["acts"])
        return "auto"

    def detach(self) -> None:
        self.agent.off("syllable", self._on_syllable)
        self.agent.remove_tick_hook(self._on_tick)


def feel(agent: Agent, kind: str, args: list[str] | tuple[str, ...] = (), tracker: ReplyTracker | None = None) -> str:
    """The ``/pleasure`` and ``/pain`` commands: ``[amount 0-1] [for: speech|move|now]``.
    Without a "for", it is for its answer to what you last said if that came
    in the last few seconds (``tracker``), else for its last words or movement."""
    amount = 0.6 if kind == "pleasure" else 0.5
    credit: Any = None
    for a in args:
        a = a.lower()
        try:
            amount = float(a)
            continue
        except ValueError:
            pass
        if a not in _FOR:
            return f"/{kind} [0-1] [speech | move | now]"
        credit = _FOR[a]
    if not 0.0 <= amount <= 1.0:
        return f"the amount is between 0 and 1 (e.g. /{kind} 0.5)"
    if credit is None:
        credit = tracker.credit() if tracker is not None else "auto"
    agent.feel(kind, amount, credit, "you")
    return ""


def describe_feeling(e: dict[str, Any]) -> str:
    """One line about a ``pleasure`` / ``pain`` event: what it felt and what it recalled."""
    kind = e.get("kind", "")
    x = float(e.get("intensity", 0.0))
    rec = e.get("recalled") or []
    what = ""
    if rec:
        ago = e.get("seconds_ago")
        when = f", {ago:.1f} s ago" if isinstance(ago, (int, float)) else ""
        what = f" — it recalls \"{' '.join(rec)}\"{when}"
    elif e.get("credit") == "now":
        what = " — for whatever it is doing right now"
    else:
        what = " — it had not said or done anything just now"
    who = str(e.get("source", ""))
    by = "" if who in ("", "you") else f" (from {who})"
    return f"(anima feels {kind} {x:.1f}{what}){by}"


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
                 out: Callable[[str], None] | None = None, belt: Any = None, mind: Any = None,
                 show_thoughts: bool = False) -> None:
        self.agent = agent
        self.brain_path = brain_path
        self.show_events = show_events
        self.belt = belt
        self.mind = mind
        self.show_thoughts = show_thoughts and mind is not None
        self._q: queue.Queue = queue.Queue()
        self._print_lock = threading.Lock()
        self._out = out or self._default_out
        agent.on("utterance", lambda e: self._q.put(("utterance", e)))
        agent.on("stage", lambda e: self._q.put(("stage", e)))
        agent.on("ate", lambda e: self._q.put(("ate", e)))
        agent.on("tool_result", lambda e: self._q.put(("tool_result", e)))
        agent.on("thought", lambda e: self._q.put(("thought", e)))
        agent.on("pleasure", lambda e: self._q.put(("feeling", e)))
        agent.on("pain", lambda e: self._q.put(("feeling", e)))
        self.replies = ReplyTracker(agent)
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
            elif kind == "feeling":
                color = "32" if e.get("kind") == "pleasure" else "31"
                self._out(f"\033[{color}m      {describe_feeling(e)}\033[0m")
            elif kind == "tool_result" and self.show_events and not self.show_thoughts:
                self._out(f"\033[2m      (anima used {e['tool']}: {e['summary']})\033[0m")
            elif kind == "thought" and self.show_thoughts and e.get("type") not in ("speech", "social"):
                from ..mind import format_thought
                self._out("      " + format_thought({**e, "kind": e.get("type", "")}, color=True))

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
        elif cmd in ("/pleasure", "/good", "/+"):
            msg = feel(a, "pleasure", parts[1:], self.replies)
            if msg:
                self._out(msg)
        elif cmd in ("/pain", "/bad", "/-"):
            msg = feel(a, "pain", parts[1:], self.replies)
            if msg:
                self._out(msg)
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
        elif cmd == "/thoughts":
            if self.mind is None:
                self._out("its thoughts are not being read (started with --no-thoughts)")
            else:
                from ..mind import format_thought
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
                items = self.mind.recent(n)
                for t in items:
                    self._out("  " + format_thought(t, color=True))
                if not items:
                    self._out("  (nothing yet)")
        elif cmd == "/mind":
            if self.mind is None:
                self._out("its thoughts are not being read (started with --no-thoughts)")
            else:
                self.show_thoughts = not self.show_thoughts
                self._out(f"thoughts {'shown as they happen' if self.show_thoughts else 'hidden'}")
        elif cmd == "/tools":
            self._out(self.belt.report() if self.belt is not None else
                      "no programs (start with --tools FILE, see examples/tools.toml)")
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
        self._out("Talk to it whenever you like; + or - after its answer gives it pleasure or pain "
                  "(it learns from them). /help for commands, /quit to leave.\n")
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
                if line.strip() in SHORTCUTS:
                    kind, amount = SHORTCUTS[line.strip()]
                    a.feel(kind, amount, self.replies.credit(), "you")
                    continue
                if line.startswith("/"):
                    if not self.command(line):
                        break
                    continue
                a.hear(line)
                self.replies.heard(line)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            a.stop()
            if self.brain_path is not None:
                p = a.save(self.brain_path)
                print(f"\nsaved {p} — run again to continue its life.")


def run_chat(agent: Agent, brain_path: Path | None, speed: float = 1.0,
             show_events: bool = True, belt: Any = None, mind: Any = None, show_thoughts: bool = False,
             **_: Any) -> None:
    ChatUI(agent, brain_path, show_events, belt=belt, mind=mind, show_thoughts=show_thoughts).run(speed=speed)
