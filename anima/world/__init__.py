"""Worlds an anima agent can live in."""

from .base import EmptyWorld, World
from .foraging import ForagingWorld

WORLDS = {"foraging": ForagingWorld, "void": EmptyWorld}

__all__ = ["World", "EmptyWorld", "ForagingWorld", "WORLDS"]
