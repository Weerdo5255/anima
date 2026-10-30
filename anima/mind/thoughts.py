"""A log of the creature's thoughts, read out of its own brain.

The creature has no words for what goes on inside it, so its "thoughts" are
not sentences it composes.  They are what its brain is busy with from moment
to moment, decoded from the same activity that drives its behaviour — the
way a neuroscientist reads intentions out of motor cortex — and written down
as short first-person lines.

Which of the many things going on gets written down is decided the way
global-workspace accounts of consciousness describe it (Baars 1988; Dehaene
& Changeux 2011): readouts *compete* for a single workspace.  Every 100 ms
each readout may offer a candidate with a salience; the most salient one
gets in, and the workspace then stays closed for a moment (an "attentional
blink", 2.5 s awake by default).  Candidates that never win fade within a
couple of seconds and are counted as ``unconscious``.  What was just thought
habituates, so a smell that keeps flipping sides or a babble that keeps
repeating does not drown out everything else.  A few things always get
through: waking and falling asleep, someone speaking, eating, a program it
used and its answer, and the occasional look back over its memories.

Every line keeps the numbers it came from (``data``) and its salience, so
the log can be checked against behaviour (``anima experiment programs``
does: do its inclinations come true, does it mouth what it heard).

Kinds, what each reads, and an example:

* ``feeling`` — a drive (hunger, fatigue, boredom, loneliness) entering a
  new band: *I'm getting hungry (0.52)*
* ``perception`` — the smell's direction and strength, walls ahead, where
  the others are (with a kin sense), a feast too big to eat alone:
  *a clear smell of food, to my left*, *someone nearby, ahead*
* ``intention`` — the synaptic push its senses, body and cortex give each
  movement or hand, which biases the fast race between them:
  *I'm inclined to turn left*, *I feel like using snack (hand 2)*
* ``inner`` — its articulators following a sound while it listens (the
  mirror map, with the voice gated off): *I mouth "ba" silently*
* ``speech`` — its own utterances: *I say "ba bo"*
* ``action`` — eating (alone, before a rival, or a feast shared with others),
  bumping, reaching an empty hand: *I ate*, *a feast, shared with Bo*
* ``evaluation`` — phasic dopamine (reward prediction error) and its cause:
  *snack: better than I expected (+0.42)*, *things are looking up (+0.25)*
* ``surprise`` — cerebellar prediction error: *that surprised me (0.71)*
* ``memory`` — a moment stored by the hippocampus:
  *I'll remember this: going ahead, saying "ma" (+0.85)*
* ``reflection`` — what the memories stored since the last look add up to:
  *looking back since 09:40: using snack went well (5x, +0.60)*
* ``dream`` — an episode replayed in slow-wave sleep:
  *I dream of turning left (from 09:12) — it felt good (+0.60)*
* ``state`` — sleep stages: *I'm falling asleep*
* ``social`` — someone talking, smiling, frowning; another creature talking or
  getting to the food first: *someone says "ma ma"*, *Bo says "ba"*,
  *Bo got to the food first*
* ``program`` — a program it used, and the answer:
  *I use snack (with hand 2) → 'nyam', reward +0.60*

The log is JSON lines (``<brain>.thoughts.jsonl``, rotated when large) and
the latest thoughts are kept in memory for the chat, the dashboard and
``anima serve``.
"""

from __future__ import annotations

import json
import math
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

KINDS = ("feeling", "perception", "intention", "inner", "speech", "action", "evaluation", "surprise",
         "memory", "reflection", "dream", "state", "social", "program")

_BANDS = (0.30, 0.55, 0.75)  # drive level boundaries
_FEEL = {
    "hunger": ("I'm not hungry", "I'm getting hungry", "I'm hungry", "I'm very hungry"),
    "fatigue": ("I'm wide awake", "I'm getting tired", "I'm tired", "I'm very sleepy"),
    "boredom": ("I'm not bored", "I'm getting bored", "I'm bored", "I'm very bored"),
    "loneliness": ("I don't feel alone", "I'm getting lonely", "I'm lonely", "I'm very lonely"),
}
_EASING = {"hunger": "less hungry now", "fatigue": "less tired now", "boredom": "less bored now",
           "loneliness": "less lonely now"}
_CAUSE = {"food": "the food", "pain": "that bump", "smell": "getting closer to food", "social": "the company"}
_MOVES = {"forward": "go ahead", "left": "turn left", "right": "turn right"}
_MOVING = {"forward": "going ahead", "left": "turning left", "right": "turning right"}
_TOWARD = {"left": "to my left", "right": "to my right", "forward": "ahead"}

#: how clearly one channel's drive must lead (relative to its size) to count as an inclination
INTENTION_MARGIN = 0.06
URGE_MARGIN = 0.06
# habituation time constant (s) per family of thoughts: right after a thought
# of the same family its salience is near zero, recovering as 1 - exp(-dt/tau)
_HABITUATION = {"feeling": 20.0, "smell": 12.0, "wall": 15.0, "intention": 5.0, "urge": 10.0, "inner": 8.0,
                "speech": 8.0, "bump": 8.0, "reach": 12.0, "eval+": 5.0, "eval-": 5.0,
                "surprise": 8.0, "memory": 6.0, "dream": 6.0, "talk": 10.0, "kin": 15.0, "missed": 8.0,
                "feast": 15.0}


def thoughts_path(brain_path: str | Path) -> Path:
    """Where the thought log of a saved creature lives (next to its brain file)."""
    p = Path(brain_path)
    return p.with_name(p.name.removesuffix(".npz") + ".thoughts.jsonl")


def _band(x: float) -> int:
    return sum(x >= b for b in _BANDS)


def _direction(code: np.ndarray, labels: list[str]) -> tuple[str, float, float] | None:
    """(side, bearing in degrees, strength) of a smell population code."""
    total = float(code.sum())
    if total < 0.15:
        return None
    angles = np.radians([float(x) for x in labels])
    z = complex(np.sum(code * np.exp(1j * angles)))
    if abs(z) / total < 0.25:
        return ("all around", 0.0, total)
    b = math.degrees(math.atan2(z.imag, z.real))
    side = ("ahead" if abs(b) <= 25 else "behind me" if abs(b) >= 155 else
            "to my left" if b > 0 else "to my right")
    return (side, b, total)


class ThoughtStream:
    """Watches a living creature and keeps a log of its thoughts.

    It hooks into the creature (events and a tick hook) and does its work on
    the creature's own thread; ``recent()``, ``subscribe()`` and the
    ``thought`` events are safe to use from anywhere.

    path: JSON-lines file to append to (None: keep in memory only).
    blink_s / sleep_blink_s: how long the workspace stays closed after an
        ordinary thought, awake / asleep (sets the pace of the stream).
    reflect_every_s: how often (awake, simulated seconds) it looks back over
        the memories it stored since the last look (0: never).
    tool_names: hand index -> program name, so reaches read as program use.
    """

    def __init__(self, agent: Any, path: str | Path | None = None, every_ms: float = 100.0,
                 keep: int = 1000, blink_s: float = 2.5, sleep_blink_s: float = 6.0,
                 fade_s: float = 2.0, min_salience: float = 0.15, reflect_every_s: float = 180.0,
                 max_file_mb: float = 20.0, tool_names: dict[int, str] | None = None) -> None:
        self.agent = agent
        self.path = Path(path) if path else None
        self.every_ms = every_ms
        self.blink_s = blink_s
        self.sleep_blink_s = sleep_blink_s
        self.fade_s = fade_s
        self.min_salience = min_salience
        self.reflect_every_s = reflect_every_s
        self.max_bytes = int(max_file_mb * 1e6)
        self.tool_names = dict(tool_names or {})
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._file = None
        self._unflushed = 0
        self._last_flush = agent.t_ms / 1000.0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        self.counts: dict[str, int] = {k: 0 for k in KINDS}
        self.unconscious = 0  # candidates that faded without getting in
        # the workspace
        now = agent.t_ms / 1000.0
        self._candidates: dict[str, tuple[float, float, str, str, dict[str, Any], Callable | None]] = {}
        self._free_at = -1e9
        self._seen: dict[str, float] = {}  # family -> when one last got in
        self._next_sample = 0.0
        self._reflect_t = now
        self._awake_since_reflect = 0.0
        # what the readouts last reported
        self._bands = {k: _band(v) for k, v in agent.brain.drives.as_dict().items()}
        self._smell: str | None = None
        self._smell_t = -1e9
        self._drive_projs: dict[str, list[Any]] = {}
        self._inclined: str | None = None
        self._inclined_t = -1e9
        self._urge: int | None = None
        self._urge_t = -1e9
        self._inner: list[str] = []
        self._inner_cons: str | None = None
        self._heard_until = -1e9
        self._last_heard = ""
        self._wall_t = -1e9
        self._calls: dict[Any, tuple[float, str, dict[str, Any]]] = {}  # program calls awaiting an answer
        has = agent.brain.outputs
        self._motor = "motor" if "motor" in has else None
        self._hands = "hands" if "hands" in has and getattr(agent.body, "hands", None) is not None else None
        self._vocal = all(k in has for k in ("vocal.C", "vocal.V"))
        self._bearing = "sense.bearing" if "sense.bearing" in agent.brain.inputs else None
        self._kin = "sense.kin" if "sense.kin" in agent.brain.inputs else None
        self._near: str | None = None
        self._events = ("stage", "replay", "memory", "ate", "bump", "heard", "utterance", "tool_call",
                        "tool_result", "hand", "missed", "feast_alone")
        for kind in self._events:
            agent.on(kind, self._on_event)
        agent.add_tick_hook(self._on_tick)
        self._closed = False

    # ------------------------------------------------------------ public
    def recent(self, n: int = 50, kinds: tuple[str, ...] | None = None, since: float | None = None
               ) -> list[dict[str, Any]]:
        """The latest thoughts (newest last), optionally of some kinds / after a time."""
        with self._lock:
            items = list(self._recent)
        if kinds:
            items = [x for x in items if x["kind"] in kinds]
        if since is not None:
            items = [x for x in items if x["t"] > since]
        return items[-n:] if n else items

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """Call ``fn(thought)`` for every new thought (on the creature thread)."""
        self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def stats(self) -> dict[str, Any]:
        return {"counts": {k: v for k, v in self.counts.items() if v}, "total": sum(self.counts.values()),
                "unconscious": self.unconscious, "path": str(self.path) if self.path else None}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for kind in self._events:
            self.agent.off(kind, self._on_event)
        self.agent.remove_tick_hook(self._on_tick)
        if self._file is not None:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------ writing
    def think(self, kind: str, text: str, data: dict[str, Any] | None = None, salience: float = 1.0) -> None:
        """Write a thought now, bypassing the competition (creature thread)."""
        a = self.agent
        now = a.t_ms / 1000.0
        rec = {"t": round(now, 2), "clock": a.brain.drives.clock_string(), "kind": kind, "text": text,
               "salience": round(float(salience), 3), "data": data or {}}
        with self._lock:
            self._recent.append(rec)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self._file is not None:
            self._file.write(json.dumps(rec, default=_jsonable) + "\n")
            self._unflushed += 1
            if self._unflushed >= 20:
                self._flush()
        for fn in list(self._subscribers):
            fn(rec)
        a._emit("thought", {**rec, "type": kind})  # an event's own "kind" is "thought"

    def offer(self, key: str, kind: str, text: str, data: dict[str, Any] | None = None,
              salience: float = 0.5, on_admit: Callable[[], None] | None = None) -> None:
        """Put a candidate thought up for the workspace.  A newer offer with
        the same ``key`` replaces an older one; the family (``key`` up to the
        first ":") sets its habituation."""
        self._candidates[key] = (float(salience), self.agent.t_ms / 1000.0, kind, text, data or {}, on_admit)

    def _admit(self, now: float) -> None:
        for key in [k for k, c in self._candidates.items() if now - c[1] > self.fade_s]:
            del self._candidates[key]
            self.unconscious += 1
        if not self._candidates or now < self._free_at:
            return
        best, best_score = None, self.min_salience
        for key, c in self._candidates.items():
            fam = key.split(":", 1)[0]
            tau = _HABITUATION.get(fam, 5.0)
            score = c[0] * (1.0 - math.exp(-(now - self._seen.get(fam, -1e9)) / tau))
            if score > best_score:
                best, best_score = key, score
        if best is None:
            return
        salience, _, kind, text, data, on_admit = self._candidates.pop(best)
        self._seen[best.split(":", 1)[0]] = now
        self._free_at = now + (self.blink_s if self.agent.brain.awake else self.sleep_blink_s)
        if on_admit is not None:
            on_admit()
        self.think(kind, text, data, salience=best_score)

    def _flush(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._unflushed = 0
        self._last_flush = self.agent.t_ms / 1000.0
        try:
            if self.path is not None and self.path.stat().st_size > self.max_bytes:
                self._file.close()
                self.path.replace(self.path.with_name(self.path.name + ".1"))
                self._file = open(self.path, "a", encoding="utf-8")
        except OSError:  # pragma: no cover
            pass

    # ------------------------------------------------------------ events
    def _on_event(self, e: dict[str, Any]) -> None:
        k = e.get("kind")
        b = self.agent.brain
        now = self.agent.t_ms / 1000.0
        if k == "stage":
            st, fr = e.get("stage"), e.get("from")
            text = {"nrem": "deep sleep again" if fr == "rem" else "I'm falling asleep",
                    "rem": "I'm dreaming (REM sleep)", "awake": "I wake up"}.get(st, st)
            self._candidates.clear()
            self.think("state", text, {"stage": st, "from": fr})
        elif k == "replay":
            r = float(e.get("reward", 0.0))
            self.offer(f"dream:{e.get('episode_t')}", "dream", self._dream_text(e),
                       {"reward": round(r, 3), "actions": e.get("actions", {}), "episode_t": e.get("episode_t")},
                       salience=0.5 + 0.3 * min(1.0, abs(r)))
        elif k == "memory":
            r = float(e.get("reward", 0.0))
            what = self._actions_text(e.get("actions", {})) or "this moment"
            text = (f"I'll remember this: {what} ({r:+.2f})" if abs(r) >= 0.05
                    else f"I'll remember this: {what} (it surprised me)")
            self.offer("memory", "memory", text, {"reward": round(r, 3), "actions": e.get("actions", {}),
                                                   "surprise": round(float(e.get("surprise", 0.0)), 3)},
                       salience=0.4 + 0.4 * min(1.0, abs(r)))
        elif k == "ate":
            data = {"hunger": round(float(b.drives.hunger), 3)}
            if e.get("feast"):
                who = " and ".join(e.get("with", [])) or "the others"
                self.think("action", f"a feast, shared with {who}", {**data, "feast": True,
                                                                   "with": e.get("with", [])})
            elif e.get("beat"):
                self.think("action", f"I ate — before {' and '.join(e['beat'])} could",
                           {**data, "beat": e["beat"]})
            else:
                self.think("action", "I ate", data)
        elif k == "missed":
            self.offer("missed", "social", f"{e.get('by', 'someone')} got to the food first",
                       {"by": e.get("by")}, salience=0.6)
        elif k == "feast_alone":
            self.offer("feast", "perception", "a feast — but it is too big to eat alone", {}, salience=0.65)
        elif k == "bump":
            self.offer("bump", "action", "ouch — I bumped into a wall", {}, salience=0.6)
        elif k == "heard":
            from ..body.voice import parse_social_signals
            text = str(e.get("text", ""))
            val = float(e.get("valence", 0.0))
            spoken, _ = parse_social_signals(text)  # what is left once smiles and frowns are taken out
            words = " ".join(w for w in spoken.split() if any(ch.isalpha() for ch in w))
            who = e.get("from")
            if words and who:  # another creature: one voice among others, it competes for attention
                self._listen(words, now)
                self.offer("talk", "social", f"{who} says \"{words[:40]}\"",
                           {"text": text[:80], "from": who, "loudness": e.get("loudness")},
                           salience=0.35 + 0.3 * float(e.get("loudness") or 1.0))
            elif words:
                self._listen(words, now)
                self.think("social", f"someone says \"{words[:40]}\"", {"text": text[:80]})
            if val > 0:
                self.think("social", "someone smiles at me", {"valence": val})
            elif val < 0:
                self.think("social", "someone frowns at me", {"valence": val})
        elif k == "utterance":
            self.offer("speech", "speech", f"I say \"{e.get('text', '')}\"", {"text": e.get("text", "")},
                       salience=0.4)
        elif k == "hand":
            if int(e["hand"]) not in self.tool_names:
                self.offer("reach", "action", f"I reach with hand {e['hand']} (nothing there)",
                           {"hand": e["hand"]}, salience=0.3)
        elif k == "tool_call":
            # written together with the answer if that comes within a second
            how = (f"by saying \"{e.get('what')}\"" if e.get("how") == "say" else f"with hand {e.get('what')}")
            self._calls[e.get("call_id")] = (now, f"I use {e.get('tool')} ({how})",
                                             {"tool": e.get("tool"), "call_id": e.get("call_id"),
                                              "how": e.get("how"), "what": e.get("what")})
        elif k == "tool_result":
            if e.get("say"):
                self._listen(str(e["say"]), now)
            answer = {"ok": e.get("ok"), "reward": e.get("reward"), "say": e.get("say"), "food": e.get("food")}
            call = self._calls.pop(e.get("call_id"), None)
            if call is not None:
                self.think("program", f"{call[1]} → {e.get('summary', '')}", {**call[2], **answer})
            else:
                self.think("program", f"{e.get('tool')} answers: {e.get('summary', '')}",
                           {"tool": e.get("tool"), "call_id": e.get("call_id"), **answer})

    def _listen(self, words: str, now: float) -> None:
        self._last_heard = words
        self._heard_until = now + 1.5
        self._inner, self._inner_cons = [], None

    def _action_parts(self, actions: dict[str, Any]) -> list[str]:
        parts = []
        a = self.agent
        if "motor" in actions and self._motor and a.body.motor is not None:
            lab = a.body.motor.labels[int(actions["motor"])]
            parts.append(_MOVING.get(lab, lab))
        if "vocal.V" in actions and self._vocal and a.body.voice is not None:
            c = int(actions.get("vocal.C", -1))
            syl = (a.body.voice.cons_labels[c] if c >= 0 else "") + a.body.voice.vow_labels[int(actions["vocal.V"])]
            parts.append(f"saying \"{syl}\"")
        if "hands" in actions:
            h = int(actions["hands"])
            parts.append(f"using {self.tool_names[h]}" if h in self.tool_names else f"reaching with hand {h}")
        return parts

    def _actions_text(self, actions: dict[str, Any]) -> str:
        return ", ".join(self._action_parts(actions))

    def _when(self, t_s: float) -> str:
        d = self.agent.brain.drives
        minutes = int(((d.cfg.start_phase + float(t_s) / d.cfg.day_length_s) % 1.0) * 24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def _dream_text(self, e: dict[str, Any]) -> str:
        what = self._actions_text(e.get("actions", {})) or "a moment"
        r = float(e.get("reward", 0.0))
        t0 = e.get("episode_t")
        when = f" (from {self._when(t0)})" if t0 is not None else ""
        feel = "good" if r > 0.05 else "bad" if r < -0.05 else "strange"
        return f"I dream of {what}{when} — it felt {feel} ({r:+.2f})"

    # ------------------------------------------------------------ sampling
    def _on_tick(self, agent: Any) -> None:
        now = agent.t_ms / 1000.0
        if now < self._next_sample:
            return
        self._next_sample = now + self.every_ms / 1000.0
        b = agent.brain
        if b.awake:
            self._feelings(b)
            if self._bearing:
                self._perceive(b, now)
            if self._motor:
                self._intend(b, now)
            if self._hands:
                self._feel_like(b, now)
            if self._vocal:
                self._inner_speech(b, now)
            self._evaluate(b)
            self._surprise(b)
            self._awake_since_reflect += self.every_ms / 1000.0
            if self.reflect_every_s and self._awake_since_reflect >= self.reflect_every_s:
                self._reflect(b, now)
        for cid in [c for c, (t, _, _) in self._calls.items() if now - t >= 1.0]:
            _, text, data = self._calls.pop(cid)
            self.think("program", text + " …", data)
        self._admit(now)
        if self._file is not None and self._unflushed and now - self._last_flush >= 5.0:
            self._flush()

    def _feelings(self, b: Any) -> None:
        for name, level in b.drives.as_dict().items():
            band = _band(level)
            old = self._bands.get(name, band)
            if band == old:
                self._candidates.pop(f"feeling:{name}", None)
                continue

            def admit(name=name, band=band) -> None:
                self._bands[name] = band
            text = _FEEL[name][band] if band > old or band == 0 else _EASING[name]
            self.offer(f"feeling:{name}", "feeling", f"{text} ({level:.2f})",
                       {"drive": name, "level": round(level, 3), "band": band},
                       salience=(0.45 + 0.15 * band) if band > old else 0.4, on_admit=admit)

    def _perceive(self, b: Any, now: float) -> None:
        code = np.asarray(b.get_input(self._bearing), dtype=float)
        d = _direction(code, b.inputs[self._bearing].labels)
        side = None if d is None else d[0]
        if side != self._smell:
            def admit(side=side) -> None:
                self._smell = side
            if side is None:
                self.offer("smell", "perception", "I can't smell any food", {}, salience=0.3, on_admit=admit)
            else:
                strength = "faint" if d[2] < 0.6 else "clear" if d[2] < 1.5 else "strong"
                self.offer("smell", "perception", f"a {strength} smell of food, {side}",
                           {"side": side, "bearing": round(d[1], 1), "strength": round(d[2], 3)},
                           salience=0.3 + (0.15 if strength == "strong" else 0.0), on_admit=admit)
        else:
            self._candidates.pop("smell", None)
        if self._kin:
            kd = _direction(np.asarray(b.get_input(self._kin), dtype=float), b.inputs[self._kin].labels)
            near = None if kd is None or kd[2] < 2.0 else kd[0]  # 2.0: another creature within ~7 units
            if near != self._near:
                def admit_kin(near=near) -> None:
                    self._near = near
                if near is None:
                    self.offer("kin", "perception", "nobody near me now", {}, salience=0.25, on_admit=admit_kin)
                else:
                    where = "all around me" if near == "all around" else near
                    self.offer("kin", "perception", f"someone nearby, {where}",
                               {"side": near, "bearing": round(kd[1], 1), "strength": round(kd[2], 3)},
                               salience=0.3, on_admit=admit_kin)
            else:
                self._candidates.pop("kin", None)
        if "sense.wall" in b.inputs:
            wall = b.get_input("sense.wall")
            if len(wall) > 1 and wall[1] > 0.6 and now - self._wall_t > 4.0:
                self._wall_t = now
                self.offer("wall", "perception", "a wall right in front of me",
                           {"proximity": round(float(wall[1]), 2)}, salience=0.45)

    def _drive(self, group: str) -> np.ndarray:
        """The push each channel of an output group gets from the rest of the
        brain: presynaptic firing rates (a ~1 s average) through the current
        weights, its own recurrent and inhibitory pools left out.  This is
        what biases the fast, noisy race between the channels — the
        "intention" a decoder can read before the race is run."""
        net = self.agent.brain.net
        projs = self._drive_projs.get(group)
        if projs is None:
            projs = [p for p in net.projections.values() if p.post == group and not p.pre.startswith(group)]
            self._drive_projs[group] = projs
        pop = net.populations[group]
        total = net.drive(projs, group)
        return total.reshape(pop.channels, pop.channel_size).mean(axis=1)

    @staticmethod
    def _lead(d: np.ndarray) -> tuple[int, float]:
        order = np.argsort(d)
        lead, second = float(d[order[-1]]), float(d[order[-2]])
        return int(order[-1]), (lead - second) / max(abs(lead), 1e-9)

    def _intend(self, b: Any, now: float) -> None:
        labels = self.agent.body.motor.labels if self.agent.body.motor is not None else []
        if not labels:
            return
        k, margin = self._lead(self._drive(self._motor))
        if margin < INTENTION_MARGIN:
            self._candidates.pop("intention", None)
            if now - self._inclined_t > 5.0:
                self._inclined = None  # no clear inclination for a while
            return
        lab = labels[k]
        if lab == self._inclined:
            self._inclined_t = now
            self._candidates.pop("intention", None)
            return
        toward = self._smell is not None and _TOWARD.get(lab) == self._smell
        why = ", toward the smell" if toward else ""

        def admit(lab=lab) -> None:
            self._inclined, self._inclined_t = lab, self.agent.t_ms / 1000.0
        self.offer("intention", "intention", f"I'm inclined to {_MOVES.get(lab, lab)}{why}",
                   {"move": lab, "margin": round(margin, 3)},
                   salience=0.25 + min(0.3, 2.0 * (margin - INTENTION_MARGIN)) + (0.1 if toward else 0.0),
                   on_admit=admit)

    def _feel_like(self, b: Any, now: float) -> None:
        """Which hand (program) the brain is pushing toward."""
        k, margin = self._lead(self._drive(self._hands))
        if margin < URGE_MARGIN:
            self._candidates.pop("urge", None)
            if now - self._urge_t > 10.0:
                self._urge = None
            return
        if k == self._urge:
            self._urge_t = now
            self._candidates.pop("urge", None)
            return
        what = f"using {self.tool_names[k]} (hand {k})" if k in self.tool_names else f"reaching with hand {k}"

        def admit(k=k) -> None:
            self._urge, self._urge_t = k, self.agent.t_ms / 1000.0
        self.offer("urge", "intention", f"I feel like {what}",
                   {"hand": k, "program": self.tool_names.get(k), "margin": round(margin, 3)},
                   salience=0.3 + min(0.3, 2.0 * (margin - URGE_MARGIN)), on_admit=admit)

    def _inner_speech(self, b: Any, now: float) -> None:
        voice, ear = self.agent.body.voice, self.agent.body.ear
        if ear is not None and ear.external_pending:
            self._heard_until = now + 1.0  # the echoic trace outlasts the sound by about a second
        listening = now <= self._heard_until
        if listening and voice is not None and not voice.speaking:
            c = np.asarray(b.channel_activity("vocal.C"), dtype=float)
            sc = np.sort(c)
            if sc[-1] >= 0.3 and sc[-1] >= 1.3 * sc[-2]:
                cons = voice.cons_labels[int(np.argmax(c))]
                if cons != self._inner_cons and len(self._inner) < 4:
                    v = np.asarray(b.channel_activity("vocal.V"), dtype=float)
                    self._inner.append(cons + voice.vow_labels[int(np.argmax(v))])
                    self._inner_cons = cons
        if self._inner and not listening:
            syl = " ".join(self._inner)
            self.offer("inner", "inner", f"I mouth \"{syl}\" silently (I heard \"{self._last_heard[:30]}\")",
                       {"syllables": list(self._inner), "heard": self._last_heard}, salience=0.6)
            self._inner, self._inner_cons = [], None

    def _evaluate(self, b: Any) -> None:
        da = float(b.mods.dopamine)  # the phasic, signed part (tonic is separate)
        if abs(da) <= 0.2:
            return
        # what brought it: the biggest reward of the last half second, if any
        recent: dict[str, float] = defaultdict(float)
        for t, amount, source in reversed(b.reward_log):
            if b.time_s - t > 0.5:
                break
            recent[source] += amount
        cause = max(recent.items(), key=lambda kv: abs(kv[1]), default=("", 0.0))
        if abs(cause[1]) >= 0.05 and (cause[1] > 0) == (da > 0):
            src = cause[0]
            what = _CAUSE.get(src, src.split(":", 1)[1] if src.startswith("program:") else src)
            text = (f"{what}: better than I expected ({da:+.2f})" if da > 0
                    else f"{what}: worse than I expected ({da:+.2f})")
        else:  # no reward: the critic's forecast itself moved (a temporal-difference error)
            src = ""
            text = f"things are looking up ({da:+.2f})" if da > 0 else f"things are looking worse ({da:+.2f})"
        self.offer("eval+" if da > 0 else "eval-", "evaluation", text,
                   {"phasic_dopamine": round(da, 3), "cause": src}, salience=min(1.0, 0.3 + abs(da)))
    def _surprise(self, b: Any) -> None:
        s = float(b.cerebellum.surprise)
        if s > 0.6:
            self.offer("surprise", "surprise", f"that surprised me ({s:.2f})", {"surprise": round(s, 3)},
                       salience=0.5 * s)

    def _reflect(self, b: Any, now: float) -> None:
        """Look back over the memories stored since the last look."""
        since = self._reflect_t
        self._reflect_t, self._awake_since_reflect = now, 0.0
        eps = [e for e in b.hippocampus.episodes if e.t_s > since]
        if not eps:
            return
        groups: dict[str, list[float]] = defaultdict(list)
        for e in eps:
            for part in self._action_parts(e.actions) or ["nothing in particular"]:
                groups[part].append(float(e.reward))
        ranked = sorted(groups.items(), key=lambda kv: -abs(sum(kv[1])))[:3]
        bits, summary = [], []
        for what, rs in ranked:
            mean = sum(rs) / len(rs)
            how = "went well" if mean > 0.05 else "went badly" if mean < -0.05 else "was surprising"
            bits.append(f"{what} {how} ({len(rs)}x, {mean:+.2f})")
            summary.append({"what": what, "n": len(rs), "mean_reward": round(mean, 3)})
        self.think("reflection", f"looking back since {self._when(since)}: " + "; ".join(bits),
                   {"since": round(since, 2), "memories": len(eps), "items": summary})


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def read_thoughts(path: str | Path, n: int | None = None, kinds: tuple[str, ...] | None = None
                  ) -> list[dict[str, Any]]:
    """Read a thought log (JSON lines), optionally the last ``n`` of some kinds."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if kinds and rec.get("kind") not in kinds:
                continue
            out.append(rec)
    return out[-n:] if n else out


def format_thought(rec: dict[str, Any], color: bool = False) -> str:
    """One thought as a line of text: ``12:31  feeling    I'm getting hungry (0.52)``."""
    kind = rec.get("kind", "")
    line = f"{rec.get('clock', '--:--')}  {kind:<10} {rec.get('text', '')}"
    if not color:
        return line
    tint = {"feeling": "35", "perception": "34", "intention": "36", "inner": "33", "speech": "1;36",
            "action": "32", "evaluation": "33", "surprise": "33", "memory": "35", "reflection": "1;35",
            "dream": "2;35", "state": "2", "social": "1", "program": "1;32"}.get(kind, "0")
    return f"\033[{tint}m{line}\033[0m"


__all__ = ["ThoughtStream", "KINDS", "thoughts_path", "read_thoughts", "format_thought"]
