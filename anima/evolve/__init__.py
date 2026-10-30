"""Creatures living together, competing and cooperating — and evolving.

:class:`Colony` runs several creatures in one shared arena
(:mod:`anima.world.arena`) day after day; at the end of a day in which one
of them clearly did best, a genetic algorithm (:mod:`anima.evolve.genome`)
fills the next day with it and variations of it.  ``anima arena`` is the
command line for it.
"""

from .colony import NAMES, POLICY_FIELDS, Colony, ColonyConfig, DayStats, creature_name, decide, decide_improved
from .genome import GENE_NAMES, GENES, Gene, crossover, gene_diff, mutate, read_genes, write_genes

__all__ = ["Colony", "ColonyConfig", "DayStats", "decide", "decide_improved", "creature_name", "NAMES",
           "POLICY_FIELDS",
           "Gene", "GENES", "GENE_NAMES", "read_genes", "write_genes", "mutate", "crossover", "gene_diff"]
