"""World interface.

A world is anything that produces sensations and responds to movements.
Implement these five methods to put an anima brain in your own world
(a game, a robot, a simulator, a data stream...).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class World:
    """Base class; also usable as an empty world with nothing in it."""

    name = "void"

    def observe(self) -> dict[str, np.ndarray]:
        """Current sensor readings, each in [0, 1] (keys are mapped to brain
        input ports by the body's ``sensor_ports``)."""
        return {}

    def apply(self, action: str) -> list[tuple[str, dict[str, Any]]]:
        """Perform a movement; return events such as ``("ate", {...})``,
        ``("bump", {...})``."""
        return []

    def step(self, dt_s: float) -> None:
        """Let time pass (food regrows, things move)."""

    def smell_ahead(self) -> float:
        """Scalar 'how promising is what is ahead' used for approach reward."""
        return 0.0

    def render(self, width: int = 40, height: int = 20) -> list[str]:
        return []

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class EmptyWorld(World):
    """Nothing to see, smell or touch: the brain lives on its own dynamics."""

    name = "void"
