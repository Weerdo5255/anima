"""A colony from Python: creatures that share an arena, and a genetic algorithm.

    python examples/06_colony.py                 # 3 small creatures, two short days

The same as ``anima arena --population 3 --cortex 400 --day 300 --days 2``,
but here you can watch whatever you like.  Each creature has its own brain
and thought log; they compete for food, can share feasts that one alone
cannot eat, and hear each other.  At dawn, if one ate clearly more than the
others, it and variations of it make up the next day's colony.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anima.evolve import Colony, ColonyConfig  # noqa: E402


def main() -> None:
    where = Path(tempfile.mkdtemp()) / "colony"
    cfg = ColonyConfig(population=3, cortex=400, day_length_s=300.0,
                       delete_old=True)  # replaced creatures are deleted, not archived
    col = Colony.create(where, cfg)
    print("living in", where, "-", ", ".join(col.names))

    # what they think about each other (a small selection of their thoughts)
    def social(name: str, e: dict) -> None:
        if e["type"] in ("social", "action") and any(n in e["text"] for n in col.names):
            print(f"   {name:>4}: {e['clock']}  {e['text']}")
    col.listen("thought", social)
    col.listen("ate", lambda name, e: print(f"   {name:>4} shared a feast with {', '.join(e['with'])}")
               if e.get("feast") else None)

    try:
        col.run(days=2, report_every_h=4)
    finally:
        col.close()
    print("\nfamily tree:\n" + col.lineage())


if __name__ == "__main__":  # needed when creatures run in worker processes (workers > 1)
    main()
