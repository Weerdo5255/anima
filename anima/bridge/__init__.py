"""A local connection between a living creature and other programs.

:class:`Bridge` serves the creature over HTTP (``anima serve``);
:class:`BridgeClient` is the other end, for any Python program.
"""

from .client import BridgeClient, BridgeError
from .server import Bridge

__all__ = ["Bridge", "BridgeClient", "BridgeError"]
