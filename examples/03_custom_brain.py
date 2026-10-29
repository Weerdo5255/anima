"""Build your own brain and your own world.

This creature has a small cortex, two light sensors and two motor channels
(left/right wheel, like a Braitenberg vehicle).  Nothing tells it which
wheel to drive; dopamine released whenever it gets closer to the light
teaches it.  The same builder accepts TOML/JSON files (see ``anima spec``).

    python examples/03_custom_brain.py
"""

import math

import numpy as np

from anima import Agent, BodyConfig, BrainBuilder, World
from anima.body import MotorConfig

# ---------------------------------------------------------------- the brain
b = BrainBuilder("phototaxis", seed=2)
b.input("intero", 20, channels=4, role="intero", gated=False,
        labels=["hunger", "fatigue", "boredom", "loneliness"], noise=2.0)
b.input("eyes", 40, channels=2, labels=["left eye", "right eye"], max_current=12.0, noise=2.0)
b.population("cortex.E", 200, kind="RS", noise=4.0, jitter=1.0, role="cortex", target_rate=4.0)
b.population("cortex.I", 50, kind="FS", noise=3.5, jitter=1.0, role="cortex")
b.output("motor", 40, channels=2, labels=["veer left", "veer right"], noise=3.5, bias=1.0)
b.population("motor.inh", 10, kind="FS", noise=2.0, role="motor")
b.connect("cortex.E", "cortex.E", p=0.1, weight=0.015, rule={"type": "stdp", "gate": "acetylcholine"})
b.connect("cortex.E", "cortex.I", p=0.2, weight=0.015)
b.connect("cortex.I", "cortex.E", p=0.3, weight=0.08)
b.connect("eyes", "cortex.E", p=0.2, weight=0.025)
b.connect("eyes", "motor", p=0.5, weight=0.008, w_max=0.05, fixed_indegree=True,
          rule={"type": "reward_stdp", "a_minus": 0.0, "tau_e": 300.0, "lr": 0.08, "normalize": "both"})
b.connect("intero", "motor", p=0.5, weight=0.02, pre_channel_gain=[1, 0, 1, 0])
b.connect("cortex.E", "motor", p=0.1, weight=0.003, w_max=0.02, fixed_indegree=True,
          rule={"type": "reward_stdp", "a_minus": 0.0, "tau_e": 300.0, "lr": 0.05, "normalize": "both"})
b.connect("motor", "motor.inh", p=0.5, weight=0.02)
b.connect("motor.inh", "motor", p=0.5, weight=0.06)
b.configure("drives", day_length_s=3600.0)
brain = b.build()


# ---------------------------------------------------------------- the world
class LightWorld(World):
    name = "light"

    def __init__(self):
        self.x, self.y, self.h = 0.0, 0.0, 0.0
        self.light = (6.0, 4.0)

    def _brightness(self, side):  # each eye looks 40 degrees off the heading
        a = self.h + side * math.radians(40)
        dx, dy = self.light[0] - self.x, self.light[1] - self.y
        d = math.hypot(dx, dy)
        facing = max(0.0, math.cos(math.atan2(dy, dx) - a))
        return facing * math.exp(-d / 8.0)

    def observe(self):
        return {"eyes": np.array([self._brightness(+1), self._brightness(-1)]) * 1.5}

    def apply(self, action):
        self.h += math.radians(20) * (1 if action == "veer left" else -1)
        self.x += 0.4 * math.cos(self.h)
        self.y += 0.4 * math.sin(self.h)
        d = math.hypot(self.x - self.light[0], self.y - self.light[1])
        if d < 1.0 or d > 10.0:
            # reached it (or wandered off): a new light appears somewhere ahead
            a = self.h + float(np.random.uniform(-1.5, 1.5))
            r = float(np.random.uniform(3.0, 6.0))
            self.light = (self.x + r * math.cos(a), self.y + r * math.sin(a))
            if d < 1.0:
                return [("ate", {"energy": 0.2})]
        return []

    def smell_ahead(self):
        return self._brightness(+1) + self._brightness(-1)


agent = Agent(brain, LightWorld(), BodyConfig(sensor_ports={"eyes": "eyes"},
                                               motor=MotorConfig(threshold=0.15)))
agent.on("ate", lambda e: print(f"{e['t']:6.1f}s  reached the light"))
for _ in range(6):
    agent.run(30)
    print(f"after {agent.t_ms / 1000:.0f}s: {agent.body.stats['ate']} lights reached")
