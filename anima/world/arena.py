"""A shared arena: several creatures on one patch of land.

The physics are those of :class:`~anima.world.foraging.ForagingWorld` —
smell, whiskers, taste, steps and turns — with the others added:

* **Food is shared.**  Whoever gets to a piece first eats it; it grows back
  somewhere else.  A creature that was close by (``contest_radius``) when
  another ate learns that it *missed* it.  That is competition, and nobody
  has to decide to compete: it follows from wanting the same food.
* **Feasts need two.**  Now and then a feast appears: it smells stronger
  than ordinary food and is worth more (``feast_energy`` *each*), but one
  creature alone cannot eat it.  When two or more are within
  ``feast_radius`` of it at the same time, all of them eat.  Whether to go
  for it, and to stay near it until someone else comes, is theirs to learn —
  the stag hunt of game theory (Skyrms 2004): cooperating pays more, but
  only if the other one comes too.
* **Food nobody eats withers** after ``wither_s`` (one to two minutes) and
  grows back somewhere else, feasts too.  Without this, food in a corner
  nobody visits stays there for ever, nothing new grows, and creatures that
  keep to the middle starve (they did, in an early version of the colony
  experiment).
* **They can hear each other.**  What one says is heard by the others
  within ``hear_radius`` (more faintly further away); being talked to
  relieves loneliness, as when you talk to it.  The colony delivers this
  (see :mod:`anima.evolve`).
* **They can sense each other.**  ``observe()`` reports a ``kin`` bearing
  code — where the others are, like the smell code for food — which reaches
  the brain if it has a ``sense.kin`` port (``infant_spec(kin=True)``).

Each creature sees the arena through a :class:`CreatureView`, which is a
:class:`~anima.world.base.World`, so bodies and brains need no changes.  A
view saves like a ``ForagingWorld``, so a creature that lived in an arena
can be loaded and live alone.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import World
from .foraging import OFFER_PLACES

GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SharedArena:
    """The shared world.  Creatures are slots ``0..n-1`` (see :meth:`add`)."""

    name = "arena"

    def __init__(self, size: float = 18.0, n_food: int = 4, n_feasts: int = 1, seed: int | None = None,
                 smell_length: float = 6.0, step: float = 0.7, turn_deg: float = 30.0,
                 eat_radius: float = 1.0, regrow_s: tuple[float, float] = (4.0, 12.0),
                 food_energy: float = 0.25, feast_energy: float = 0.4, feast_radius: float = 1.6,
                 feast_regrow_s: tuple[float, float] = (15.0, 30.0), feast_smell: float = 2.0,
                 kin_length: float = 5.0, hear_radius: float = 7.0, contest_radius: float = 2.5,
                 n_bearings: int = 8, tuning_kappa: float = 2.5,
                 wither_s: tuple[float, float] | None = (60.0, 120.0)) -> None:
        self.size = float(size)
        self.rng = np.random.default_rng(seed)
        self.smell_length = smell_length
        self.step_len = step
        self.turn = math.radians(turn_deg)
        self.eat_radius = eat_radius
        self.regrow_s = tuple(regrow_s)
        self.food_energy = food_energy
        self.feast_energy = feast_energy
        self.feast_radius = feast_radius
        self.feast_regrow_s = tuple(feast_regrow_s)
        self.feast_smell = feast_smell
        self.kin_length = kin_length
        self.hear_radius = hear_radius
        self.contest_radius = contest_radius
        self.n_bearings = n_bearings
        self.kappa = tuning_kappa
        self.wither_s = tuple(wither_s) if wither_s else None
        self.prefs = np.linspace(-math.pi, math.pi, n_bearings, endpoint=False)
        self.food: list[list[float]] = [self._random_spot() for _ in range(n_food)]
        self.food_life: list[float] = [self._lifetime() for _ in self.food]  # seconds until it withers
        self.gift: list[bool] = [False] * len(self.food)  # placed by someone: does not grow back
        self.regrow: list[float] = []
        self.feasts: list[list[float]] = [self._random_spot() for _ in range(n_feasts)]
        self.feast_life: list[float] = [self._lifetime() for _ in self.feasts]
        self.feast_regrow: list[float] = []
        self.wild = True
        self.bodies: list[dict[str, Any]] = []
        self.pending: list[list[tuple[str, dict[str, Any]]]] = []
        self.counts = self._zero_counts()

    @staticmethod
    def _zero_counts() -> dict[str, int]:
        return {"meals": 0, "feasts": 0, "contested": 0, "withered": 0}

    def reset_counts(self) -> None:
        self.counts = self._zero_counts()

    def kwargs(self) -> dict[str, Any]:
        """The settings this arena was made with (for saving)."""
        self._sync()
        return {"size": self.size, "n_food": sum(not g for g in self.gift) + len(self.regrow),
                "n_feasts": len(self.feasts) + len(self.feast_regrow), "smell_length": self.smell_length,
                "step": self.step_len, "turn_deg": math.degrees(self.turn), "eat_radius": self.eat_radius,
                "regrow_s": list(self.regrow_s), "food_energy": self.food_energy,
                "feast_energy": self.feast_energy, "feast_radius": self.feast_radius,
                "feast_regrow_s": list(self.feast_regrow_s), "feast_smell": self.feast_smell,
                "kin_length": self.kin_length, "hear_radius": self.hear_radius,
                "contest_radius": self.contest_radius, "n_bearings": self.n_bearings, "tuning_kappa": self.kappa,
                "wither_s": list(self.wither_s) if self.wither_s else None}

    def _random_spot(self) -> list[float]:
        m = 1.5
        return [float(self.rng.uniform(m, self.size - m)), float(self.rng.uniform(m, self.size - m))]

    def _lifetime(self) -> float:
        return float(self.rng.uniform(*self.wither_s)) if self.wither_s else math.inf

    def _sync(self) -> None:
        """``food`` and ``feasts`` may be changed from outside (a test, a
        teacher appending to ``view.food``): food that appeared that way gets a
        lifetime, and is wild."""
        n = len(self.food)
        if len(self.food_life) != n or len(self.gift) != n:
            self.food_life = (self.food_life + [self._lifetime() for _ in range(n - len(self.food_life))])[:n]
            self.gift = (self.gift + [False] * (n - len(self.gift)))[:n]
        n = len(self.feasts)
        if len(self.feast_life) != n:
            self.feast_life = (self.feast_life + [self._lifetime() for _ in range(n - len(self.feast_life))])[:n]

    # ------------------------------------------------------------- creatures
    def add(self, name: str) -> int:
        """Put a new creature in a random spot; returns its slot."""
        self.bodies.append({})
        self.pending.append([])
        i = len(self.bodies) - 1
        self.reset(i, name)
        return i

    def reset(self, i: int, name: str, near: tuple[float, float] | None = None, spread: float = 2.0) -> None:
        """Give slot ``i`` to a (new) creature: at a random spot, or within
        ``spread`` of ``near`` (a child next to its parent)."""
        if near is None:
            x, y = self._random_spot()
        else:
            m = 0.5
            r, a = spread * math.sqrt(float(self.rng.uniform())), float(self.rng.uniform(0, 2 * math.pi))
            x = float(np.clip(near[0] + r * math.cos(a), m, self.size - m))
            y = float(np.clip(near[1] + r * math.sin(a), m, self.size - m))
        self.bodies[i] = {"name": name, "x": x, "y": y, "heading": float(self.rng.uniform(0, 2 * math.pi)),
                          "taste_left": 0.0, "trail": [], "at_feast": False, "eaten": 0}
        self.pending[i] = []

    def view(self, i: int) -> "CreatureView":
        return CreatureView(self, i)

    def distance(self, i: int, j: int) -> float:
        a, b = self.bodies[i], self.bodies[j]
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    def others_within(self, i: int, radius: float) -> list[int]:
        return [j for j in range(len(self.bodies)) if j != i and self.distance(i, j) <= radius]

    # ------------------------------------------------------------- sensing
    def _vectors(self, i: int, points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        b = self.bodies[i]
        if not points:
            return np.zeros(0), np.zeros(0)
        p = np.asarray(points, dtype=float)
        dx, dy = p[:, 0] - b["x"], p[:, 1] - b["y"]
        bearing = (np.arctan2(dy, dx) - b["heading"] + math.pi) % (2 * math.pi) - math.pi
        return bearing, np.hypot(dx, dy)

    def _code(self, bearing: np.ndarray, strength: np.ndarray) -> np.ndarray:
        """Population code over the bearing channels (von Mises tuning), summed over sources."""
        if len(bearing) == 0:
            return np.zeros(self.n_bearings)
        tuning = np.exp(self.kappa * (np.cos(bearing[:, None] - self.prefs[None, :]) - 1.0))
        return np.minimum(strength @ tuning, 1.0)

    def _smell(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Bearings and odour intensities of all food and feasts, from creature ``i``."""
        bf, df = self._vectors(i, self.food)
        bF, dF = self._vectors(i, self.feasts)
        bearing = np.concatenate([bf, bF])
        intensity = np.concatenate([np.exp(-df / self.smell_length),
                                    np.clip(self.feast_smell * np.exp(-dF / self.smell_length), 0.0, 1.0)])
        return bearing, intensity

    def observe(self, i: int) -> dict[str, np.ndarray]:
        b = self.bodies[i]
        bearing, intensity = self._smell(i)
        smell = self._code(bearing, np.clip(1.6 * intensity ** 0.4, 0.0, 1.0))
        others = [[o["x"], o["y"]] for j, o in enumerate(self.bodies) if j != i]
        kb, kd = self._vectors(i, others)
        kin = self._code(kb, np.clip(1.6 * np.exp(-kd / self.kin_length) ** 0.4, 0.0, 1.0)) if others \
            else np.zeros(self.n_bearings)
        walls = np.array([self._wall_proximity(i, a) for a in (self.turn * 2, 0.0, -self.turn * 2)])
        return {"bearing": smell, "wall": walls, "taste": np.array([1.0 if b["taste_left"] > 0 else 0.0]),
                "kin": kin}

    def _wall_proximity(self, i: int, rel_angle: float, reach: float = 2.5) -> float:
        b = self.bodies[i]
        a = b["heading"] + rel_angle
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for p, d in ((b["x"], dx), (b["y"], dy)):
            if d > 1e-9:
                ts.append((self.size - p) / d)
            elif d < -1e-9:
                ts.append(-p / d)
        dist = min(ts) if ts else reach
        return min(1.0, max(0.0, 1.0 - dist / reach))

    def smell_ahead(self, i: int) -> float:
        bearing, intensity = self._smell(i)
        return float(np.sum(intensity * (1.0 + np.cos(bearing)) / 2.0))

    # ------------------------------------------------------------- acting
    def apply(self, i: int, action: str) -> list[tuple[str, dict[str, Any]]]:
        """Creature ``i`` completes a movement.  Its own consequences are
        returned; consequences for the others go to their ``pending`` lists."""
        b = self.bodies[i]
        events: list[tuple[str, dict[str, Any]]] = []
        self._sync()
        if action == "left":
            b["heading"] = (b["heading"] + self.turn) % (2 * math.pi)
        elif action == "right":
            b["heading"] = (b["heading"] - self.turn) % (2 * math.pi)
        elif action == "forward":
            nx = b["x"] + self.step_len * math.cos(b["heading"])
            ny = b["y"] + self.step_len * math.sin(b["heading"])
            if 0.3 <= nx <= self.size - 0.3 and 0.3 <= ny <= self.size - 0.3:
                b["x"], b["y"] = nx, ny
                b["trail"].append((nx, ny))
                if len(b["trail"]) > 200:
                    del b["trail"][:-200]
            else:
                events.append(("bump", {"x": b["x"], "y": b["y"]}))
            events += self._eat(i)
            events += self._feast(i)
        return events

    def _eat(self, i: int) -> list[tuple[str, dict[str, Any]]]:
        b = self.bodies[i]
        for k, (fx, fy) in enumerate(self.food):
            if math.hypot(fx - b["x"], fy - b["y"]) < self.eat_radius:
                gift = self._take_food(k)
                b["taste_left"] = 0.3
                b["eaten"] += 1
                self.counts["meals"] += 1
                rivals = [j for j in self.others_within(i, self.contest_radius)]
                if rivals:
                    self.counts["contested"] += 1
                for j in rivals:
                    self.pending[j].append(("missed", {"by": b["name"]}))
                return [("ate", {"energy": self.food_energy, "gift": gift,
                                 "beat": [self.bodies[j]["name"] for j in rivals]})]
        return []

    def _take_food(self, k: int) -> bool:
        """Food item ``k`` is gone (eaten or withered); wild food grows back."""
        self.food.pop(k)
        self.food_life.pop(k)
        gift = self.gift.pop(k)
        if not gift:
            self.regrow.append(float(self.rng.uniform(*self.regrow_s)))
        return gift

    def _feast(self, i: int) -> list[tuple[str, dict[str, Any]]]:
        b = self.bodies[i]
        for k, (fx, fy) in enumerate(self.feasts):
            if math.hypot(fx - b["x"], fy - b["y"]) >= self.feast_radius:
                continue
            party = [j for j in range(len(self.bodies))
                     if math.hypot(fx - self.bodies[j]["x"], fy - self.bodies[j]["y"]) < self.feast_radius]
            if len(party) < 2:
                if b["at_feast"]:
                    return []
                b["at_feast"] = True
                return [("feast_alone", {})]
            self._take_feast(k)
            self.counts["feasts"] += 1
            names = [self.bodies[j]["name"] for j in party]
            mine: list[tuple[str, dict[str, Any]]] = []
            for j in party:
                self.bodies[j]["taste_left"] = 0.3
                self.bodies[j]["at_feast"] = False
                self.bodies[j]["eaten"] += 1
                ev = ("ate", {"energy": self.feast_energy, "gift": False, "feast": True,
                              "with": [n for n in names if n != self.bodies[j]["name"]]})
                if j == i:
                    mine.append(ev)
                else:
                    self.pending[j].append(ev)
            return mine
        b["at_feast"] = False
        return []

    def _take_feast(self, k: int) -> None:
        self.feasts.pop(k)
        self.feast_life.pop(k)
        self.feast_regrow.append(float(self.rng.uniform(*self.feast_regrow_s)))

    def step(self, dt_s: float) -> None:
        """Let time pass for the whole arena (call once per tick, not per creature)."""
        if not self.wild:
            return
        self._sync()
        if self.regrow:
            self.regrow = [t - dt_s for t in self.regrow]
            while self.regrow and min(self.regrow) <= 0:
                self.regrow.remove(min(self.regrow))
                self._grow(self._random_spot())
        if self.feast_regrow:
            self.feast_regrow = [t - dt_s for t in self.feast_regrow]
            while self.feast_regrow and min(self.feast_regrow) <= 0:
                self.feast_regrow.remove(min(self.feast_regrow))
                self.feasts.append(self._random_spot())
                self.feast_life.append(self._lifetime())
        if self.wither_s:  # (what withers now starts growing back from the next step)
            self.food_life = [t - dt_s for t in self.food_life]
            for k in [k for k, t in enumerate(self.food_life) if t <= 0][::-1]:
                self._take_food(k)
                self.counts["withered"] += 1
            self.feast_life = [t - dt_s for t in self.feast_life]
            for k in [k for k, t in enumerate(self.feast_life) if t <= 0][::-1]:
                self._take_feast(k)
                self.counts["withered"] += 1

    def _grow(self, spot: list[float], gift: bool = False) -> None:
        self.food.append(spot)
        self.food_life.append(self._lifetime())
        self.gift.append(gift)

    def step_body(self, i: int, dt_s: float) -> None:
        b = self.bodies[i]
        b["taste_left"] = max(0.0, b["taste_left"] - dt_s)

    def snapshot(self, i: int) -> dict[str, Any]:
        """Creature ``i``'s view of the arena, saved like a ForagingWorld's state
        (so the creature can be loaded on its own)."""
        b = self.bodies[i]
        self._sync()
        return {"x": b["x"], "y": b["y"], "heading": b["heading"], "food": [list(f) for f in self.food],
                "gift": list(self.gift), "regrow": list(self.regrow), "eaten": int(b["eaten"]),
                "taste_left": b["taste_left"], "trail": [list(p) for p in b["trail"]],
                "rng": self.rng.bit_generator.state}

    def take_pending(self, i: int) -> list[tuple[str, dict[str, Any]]]:
        out, self.pending[i] = self.pending[i], []
        return out

    def place_food(self, i: int, bearing_deg: float, distance: float, max_items: int = 8) -> tuple[float, float]:
        """Put a piece of food relative to creature ``i`` (a gift: it does not grow back)."""
        b = self.bodies[i]
        a = b["heading"] + math.radians(bearing_deg)
        m = 1.0
        fx = float(np.clip(b["x"] + distance * math.cos(a), m, self.size - m))
        fy = float(np.clip(b["y"] + distance * math.sin(a), m, self.size - m))
        self._sync()
        if len(self.food) < max_items:
            self._grow([fx, fy], gift=True)
        return fx, fy

    def offer_food(self, i: int, where: str = "ahead") -> tuple[float, float]:
        if where not in OFFER_PLACES:
            raise ValueError(f"where must be one of {sorted(OFFER_PLACES)}")
        spot = OFFER_PLACES[where]
        if spot is None:
            spot = (float(self.rng.uniform(-180.0, 180.0)), float(self.rng.uniform(2.5, 4.5)))
        return self.place_food(i, *spot)

    # ------------------------------------------------------------- display
    def render(self, width: int = 36, height: int = 18) -> list[str]:
        grid = [[" "] * width for _ in range(height)]

        def cell(x: float, y: float) -> tuple[int, int]:
            return (int(np.clip(x / self.size * width, 0, width - 1)),
                    int(np.clip((1 - y / self.size) * height, 0, height - 1)))

        for b in self.bodies:
            for x, y in b["trail"][-25:]:
                cx, cy = cell(x, y)
                grid[cy][cx] = "."
        for fx, fy in self.food:
            cx, cy = cell(fx, fy)
            grid[cy][cx] = "*"
        for fx, fy in self.feasts:
            cx, cy = cell(fx, fy)
            grid[cy][cx] = "@"
        for b in self.bodies:
            cx, cy = cell(b["x"], b["y"])
            grid[cy][cx] = (b["name"][:1] or "?").upper()
        top = "+" + "-" * width + "+"
        return [top] + ["|" + "".join(r) + "|" for r in grid] + [top]

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        self._sync()
        return {"kwargs": self.kwargs(), "food": [list(f) for f in self.food], "regrow": list(self.regrow),
                "food_life": [_finite(t) for t in self.food_life], "gift": list(self.gift),
                "feasts": [list(f) for f in self.feasts], "feast_regrow": list(self.feast_regrow),
                "feast_life": [_finite(t) for t in self.feast_life],
                "bodies": [{**b, "trail": [list(p) for p in b["trail"]]} for b in self.bodies],
                "counts": dict(self.counts), "rng": self.rng.bit_generator.state}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> "SharedArena":
        kw = dict(s["kwargs"])
        kw["regrow_s"], kw["feast_regrow_s"] = tuple(kw["regrow_s"]), tuple(kw["feast_regrow_s"])
        if "wither_s" in kw:  # (colonies saved before food withered: it does not)
            kw["wither_s"] = tuple(kw["wither_s"]) if kw["wither_s"] else None
        else:
            kw["wither_s"] = None
        arena = cls(**kw)
        arena.food = [list(map(float, f)) for f in s["food"]]
        arena.regrow = [float(t) for t in s["regrow"]]
        arena.food_life = [_life(t) for t in s.get("food_life", [None] * len(arena.food))]
        arena.gift = [bool(g) for g in s.get("gift", [False] * len(arena.food))]
        arena.feasts = [list(map(float, f)) for f in s["feasts"]]
        arena.feast_regrow = [float(t) for t in s["feast_regrow"]]
        arena.feast_life = [_life(t) for t in s.get("feast_life", [None] * len(arena.feasts))]
        arena.bodies = [{**b, "trail": [tuple(p) for p in b["trail"]]} for b in s["bodies"]]
        arena.pending = [[] for _ in arena.bodies]
        arena.counts = {**cls._zero_counts(), **s.get("counts", {})}
        arena.rng.bit_generator.state = s["rng"]
        return arena


def _finite(t: float) -> float | None:
    """JSON has no infinity: food that never withers is saved as null."""
    return None if math.isinf(t) else float(t)


def _life(t: float | None) -> float:
    return math.inf if t is None else float(t)


class CreatureView(World):
    """One creature's window on a :class:`SharedArena` — a World like any other.

    It calls itself "foraging" and saves like a ForagingWorld (its position,
    the food it can see), so a creature from an arena can be loaded on its
    own.  Its ``step()`` only ages its own taste; the colony steps the arena
    once per tick.
    """

    name = "foraging"

    def __init__(self, arena: SharedArena, slot: int) -> None:
        self.arena = arena
        self.slot = slot

    @property
    def body(self) -> dict[str, Any]:
        return self.arena.bodies[self.slot]

    def observe(self) -> dict[str, np.ndarray]:
        return self.arena.observe(self.slot)

    def apply(self, action: str) -> list[tuple[str, dict[str, Any]]]:
        return self.arena.apply(self.slot, action)

    def step(self, dt_s: float) -> None:
        self.arena.step_body(self.slot, dt_s)

    def smell_ahead(self) -> float:
        return self.arena.smell_ahead(self.slot)

    def render(self, width: int = 36, height: int = 18) -> list[str]:
        return self.arena.render(width, height)

    # what teachers, programs and /feed use
    @property
    def food(self) -> list[list[float]]:
        return self.arena.food

    @property
    def wild(self) -> bool:
        return self.arena.wild

    def set_wild(self, on: bool) -> None:  # the arena's food is everyone's: leave it alone
        pass

    def place_food(self, bearing_deg: float, distance: float, max_items: int = 8) -> tuple[float, float]:
        return self.arena.place_food(self.slot, bearing_deg, distance, max_items)

    def offer_food(self, where: str = "ahead", rng: np.random.Generator | None = None) -> tuple[float, float]:
        return self.arena.offer_food(self.slot, where)

    def nearest_food_distance(self) -> float:
        b = self.body
        pts = self.arena.food
        return float(min(math.hypot(f[0] - b["x"], f[1] - b["y"]) for f in pts)) if pts else float("inf")

    def state_dict(self) -> dict[str, Any]:
        return self.arena.snapshot(self.slot)

    def load_state_dict(self, state: dict[str, Any]) -> None:  # the arena keeps its own state
        pass


__all__ = ["SharedArena", "CreatureView", "GLYPHS"]
