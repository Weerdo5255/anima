"""anima — brain-like, always-on spiking agents.

anima is not a language model.  It builds a spiking neural network with
drives, neuromodulators, sleep and local plasticity, embeds it in a body and
a world, and lets it run on its own clock.  It acts, babbles and learns
whether or not anyone is talking to it.

Quick start::

    from anima import Agent
    agent = Agent.create()                 # default infant brain, foraging world
    agent.on("utterance", lambda e: print("anima:", e["text"]))
    agent.start()                          # lives in real time, in the background
    agent.hear("hello")                    # say something whenever you like

Programs it may use (``anima.tools``), the log of its thoughts
(``anima.mind``) and the local HTTP connection for other programs
(``anima.bridge``) are separate modules you attach to a living agent.
"""

from .body import Body, BodyConfig
from .brain import Brain, BrainBuilder, build_brain, infant_spec, load_spec, minimal_spec, save_spec
from .core import Network, Population, Projection
from .runtime import Agent
from .world import EmptyWorld, ForagingWorld, World

__version__ = "0.3.0"

__all__ = [
    "Agent", "Brain", "BrainBuilder", "build_brain", "infant_spec", "minimal_spec",
    "load_spec", "save_spec", "Body", "BodyConfig", "Network", "Population", "Projection",
    "World", "EmptyWorld", "ForagingWorld", "__version__",
]
