"""A small 2-D arena with food that regrows.

The agent has a position and a heading.  It smells food: each food source
emits an odour that decays with distance, and a population code over eight
egocentric bearings (like head-direction-tuned cells) reports where the
odour comes from.  Three whisker-like sensors report nearby walls.  Taste
fires when it eats.  Food is eaten automatically on arrival and regrows
elsewhere after a while.

Movements: ``forward`` (one step), ``left`` / ``right`` (turn 30 deg).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import World

#: where a teacher (or you, with ``/feed``) can put food, relative to the
#: creature: (bearing in degrees, + = left; distance).  "near" = random.
OFFER_PLACES: dict[str, tuple[float, float] | None] = {
    "ahead": (0.0, 2.0), "left": (70.0, 2.5), "right": (-70.0, 2.5), "near": None}


class ForagingWorld(World):
    name = "foraging"

    def __init__(self, size: float = 16.0, n_food: int = 3, seed: int | None = None,
                 smell_length: float = 6.0, step: float = 0.7, turn_deg: float = 30.0,
                 eat_radius: float = 1.0, regrow_s: tuple[float, float] = (4.0, 12.0),
                 food_energy: float = 0.25, n_bearings: int = 8, tuning_kappa: float = 2.5) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.smell_length = smell_length
        self.step_len = step
        self.turn = math.radians(turn_deg)
        self.eat_radius = eat_radius
        self.regrow_s = regrow_s
        self.food_energy = food_energy
        self.n_bearings = n_bearings
        self.kappa = tuning_kappa
        self.prefs = np.linspace(-math.pi, math.pi, n_bearings, endpoint=False)
        self.x = self.y = size / 2.0
        self.heading = 0.0
        self.food: list[list[float]] = [self._random_spot() for _ in range(n_food)]
        self.gift: list[bool] = [False] * n_food  # True = placed by a teacher (does not regrow)
        self.regrow: list[float] = []  # seconds until each eaten food reappears
        self.wild = True  # natural food grows back (False: only what a teacher gives)
        self.taste_left = 0.0
        self.eaten = 0
        self.trail: list[tuple[float, float]] = []

    def _random_spot(self) -> list[float]:
        m = 1.5
        return [float(self.rng.uniform(m, self.size - m)), float(self.rng.uniform(m, self.size - m))]

    # ------------------------------------------------------------- sensing
    def _food_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.food:
            return np.zeros(0), np.zeros(0)
        f = np.asarray(self.food)
        dx, dy = f[:, 0] - self.x, f[:, 1] - self.y
        dist = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - self.heading
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        intensity = np.exp(-dist / self.smell_length)
        return bearing, intensity

    def observe(self) -> dict[str, np.ndarray]:
        bearing, intensity = self._food_vectors()
        code = np.zeros(self.n_bearings)
        # Compressive (Weber-like) intensity coding: a faint odour still gives
        # a clear sense of direction, a strong one saturates.
        strength = np.clip(1.6 * intensity ** 0.4, 0.0, 1.0)
        for th, s in zip(bearing, strength):
            code += s * np.exp(self.kappa * (np.cos(th - self.prefs) - 1.0))
        code = np.clip(code, 0.0, 1.0)
        walls = np.array([self._wall_proximity(a) for a in (self.turn * 2, 0.0, -self.turn * 2)])
        return {"bearing": code, "wall": walls,
                "taste": np.array([1.0 if self.taste_left > 0 else 0.0])}

    def _wall_proximity(self, rel_angle: float, reach: float = 2.5) -> float:
        a = self.heading + rel_angle
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for p, d, lim in ((self.x, dx, self.size), (self.y, dy, self.size)):
            if d > 1e-9:
                ts.append((lim - p) / d)
            elif d < -1e-9:
                ts.append(-p / d)
        dist = min(ts) if ts else reach
        return float(np.clip(1.0 - dist / reach, 0.0, 1.0))

    def smell_ahead(self) -> float:
        bearing, intensity = self._food_vectors()
        return float(np.sum(intensity * (1.0 + np.cos(bearing)) / 2.0))

    # ------------------------------------------------------------- acting
    def apply(self, action: str) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        if action == "left":
            self.heading = (self.heading + self.turn) % (2 * math.pi)
        elif action == "right":
            self.heading = (self.heading - self.turn) % (2 * math.pi)
        elif action == "forward":
            nx = self.x + self.step_len * math.cos(self.heading)
            ny = self.y + self.step_len * math.sin(self.heading)
            if 0.3 <= nx <= self.size - 0.3 and 0.3 <= ny <= self.size - 0.3:
                self.x, self.y = nx, ny
                self.trail.append((self.x, self.y))
                if len(self.trail) > 400:
                    self.trail = self.trail[-400:]
            else:
                events.append(("bump", {"x": self.x, "y": self.y}))
            for i, (fx, fy) in enumerate(list(self.food)):
                if math.hypot(fx - self.x, fy - self.y) < self.eat_radius:
                    flags = self._gift_flags()
                    self.food.pop(i)
                    was_gift = flags.pop(i)
                    if not was_gift:  # natural food grows back elsewhere; gifts do not
                        self.regrow.append(float(self.rng.uniform(*self.regrow_s)))
                    self.taste_left = 0.3
                    self.eaten += 1
                    events.append(("ate", {"energy": self.food_energy, "gift": was_gift}))
                    break
        return events

    def place_food(self, bearing_deg: float, distance: float, max_items: int = 6) -> tuple[float, float]:
        """Put food at an egocentric bearing (deg, + = left) and distance.

        Used by teachers (scaffolding) and by the ``/feed`` command.  A gift
        is eaten once and does not grow back.  Keeps at most ``max_items``
        pieces of food: beyond that the farthest piece is replaced (a natural
        one then grows back later).  Returns the position used.
        """
        a = self.heading + math.radians(bearing_deg)
        m = 1.0
        fx = float(np.clip(self.x + distance * math.cos(a), m, self.size - m))
        fy = float(np.clip(self.y + distance * math.sin(a), m, self.size - m))
        if math.hypot(fx - self.x, fy - self.y) < self.eat_radius * 1.2:
            # pinned against a wall: put it just ahead instead
            fx = float(np.clip(self.x + 1.5 * math.cos(self.heading), m, self.size - m))
            fy = float(np.clip(self.y + 1.5 * math.sin(self.heading), m, self.size - m))
        flags = self._gift_flags()
        if len(self.food) >= max_items:
            far = int(np.argmax([math.hypot(f[0] - self.x, f[1] - self.y) for f in self.food]))
            if not flags[far]:
                self.regrow.append(float(self.rng.uniform(*self.regrow_s)))
            self.food[far] = [fx, fy]
            flags[far] = True
        else:
            self.food.append([fx, fy])
            flags.append(True)
        return fx, fy

    def offer_food(self, where: str = "ahead", rng: np.random.Generator | None = None) -> tuple[float, float]:
        """Give it a piece of food ``ahead``, to its ``left``/``right``, or ``near`` (random)."""
        if where not in OFFER_PLACES:
            raise ValueError(f"where must be one of {sorted(OFFER_PLACES)}")
        spot = OFFER_PLACES[where]
        if spot is None:
            r = rng if rng is not None else self.rng
            spot = (float(r.uniform(-180.0, 180.0)), float(r.uniform(2.5, 4.5)))
        return self.place_food(*spot)

    def _gift_flags(self) -> list[bool]:
        """``gift`` kept in step with ``food`` (which callers may assign directly)."""
        n = len(self.food)
        if len(self.gift) != n:
            self.gift = (list(self.gift) + [False] * n)[:n]
        return self.gift

    def set_wild(self, on: bool) -> None:
        """Switch natural food on or off.

        Off: the natural food is cleared away (it will grow back once switched
        on again), so the only food is what a teacher gives.
        """
        on = bool(on)
        if on == self.wild:
            return
        self.wild = on
        if not on:
            flags = self._gift_flags()
            keep = [(f, g) for f, g in zip(self.food, flags) if g]
            for _ in range(len(self.food) - len(keep)):
                self.regrow.append(float(self.rng.uniform(*self.regrow_s)))
            self.food = [f for f, _ in keep]
            self.gift = [True] * len(keep)

    def nearest_food_distance(self) -> float:
        if not self.food:
            return float("inf")
        return float(min(math.hypot(f[0] - self.x, f[1] - self.y) for f in self.food))

    def step(self, dt_s: float) -> None:
        self.taste_left = max(0.0, self.taste_left - dt_s)
        if self.regrow and self.wild:
            self.regrow = [t - dt_s for t in self.regrow]
            while self.regrow and min(self.regrow) <= 0:
                self.regrow.remove(min(self.regrow))
                self._gift_flags()
                self.food.append(self._random_spot())
                self.gift.append(False)

    # ------------------------------------------------------------- display
    def render(self, width: int = 32, height: int = 16) -> list[str]:
        grid = [[" "] * width for _ in range(height)]

        def cell(x: float, y: float) -> tuple[int, int]:
            cx = int(np.clip(x / self.size * width, 0, width - 1))
            cy = int(np.clip((1 - y / self.size) * height, 0, height - 1))
            return cx, cy

        for x, y in self.trail[-60:]:
            cx, cy = cell(x, y)
            grid[cy][cx] = "."
        for fx, fy in self.food:
            cx, cy = cell(fx, fy)
            grid[cy][cx] = "*"
        arrows = ">^<v"
        k = int(((self.heading + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))
        cx, cy = cell(self.x, self.y)
        grid[cy][cx] = arrows[k]
        top = "+" + "-" * width + "+"
        return [top] + ["|" + "".join(r) + "|" for r in grid] + [top]

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "heading": self.heading, "food": self.food,
                "gift": list(self._gift_flags()),
                "regrow": self.regrow, "eaten": self.eaten, "taste_left": self.taste_left,
                "trail": [list(p) for p in self.trail], "rng": self.rng.bit_generator.state}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.x, self.y, self.heading = float(s["x"]), float(s["y"]), float(s["heading"])
        self.food = [list(map(float, f)) for f in s["food"]]
        self.gift = [bool(g) for g in s.get("gift", [False] * len(self.food))]
        # ``wild`` is a training condition owned by a tutor (which re-applies it
        # when it resumes), so a saved world always comes back with natural food.
        self.wild = True
        self.regrow = [float(t) for t in s["regrow"]]
        self.eaten = int(s["eaten"])
        self.taste_left = float(s.get("taste_left", 0.0))
        self.trail = [(float(x), float(y)) for x, y in s.get("trail", [])]
        self.rng.bit_generator.state = s["rng"]
