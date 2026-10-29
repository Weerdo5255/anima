"""Plays a little tune.  The more bored the creature is, the more it likes it."""

import json
import random
import sys

boredom = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
tune = " ".join(random.choice(["la", "li", "lo", "lu"]) for _ in range(4))
print(json.dumps({"say": tune, "reward": round(0.5 * boredom, 3)}))
