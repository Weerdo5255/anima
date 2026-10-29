"""Says the creature's time of day.  It hears the words as sounds only."""

import sys

print(f"it is {sys.argv[1] if len(sys.argv) > 1 else 'now'}")
