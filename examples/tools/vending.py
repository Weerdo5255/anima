"""A snack machine for the creature (hand 0 in examples/tools.toml).

It reads the call (JSON on stdin) and gives food at most once every 30
creature-seconds; otherwise it just buzzes.  Its memory of the last snack is
a small file in the system's temp folder (or $ANIMA_STATE_DIR).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

call = json.load(sys.stdin)
state = Path(os.environ.get("ANIMA_STATE_DIR", tempfile.gettempdir())) / "anima_vending.json"
try:
    last = json.loads(state.read_text())["last_t"]
except (OSError, ValueError, KeyError):
    last = -1e9
now = float(call.get("t", 0.0))
if now < last:  # a new creature (or an older save): start over
    last = -1e9
if now - last >= 30.0:
    state.write_text(json.dumps({"last_t": now}))
    print(json.dumps({"food": "ahead", "say": "clunk"}))
else:
    print(json.dumps({"say": "bzz"}))
