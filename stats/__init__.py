"""Statistical analysis package for code_SNI (reviewer points R1-5 / R2-6a).

Modules
-------
:mod:`stats.long_table`
    Tidy ``dataset x mechanism x rate x method x seed x metric x value`` builder
    over the R0 per-seed aggregates, with a hard exclusion of the failed
    ``agg_baselines_deep``.
:mod:`stats.omnibus`
    Friedman test with the Iman--Davenport F correction.
:mod:`stats.posthoc`
    Nemenyi (rank-based, all-pairs) and Wilcoxon + Holm--Bonferroni
    (value-based, reference-vs-all) -- two independent routes.
:mod:`stats.effect_size`
    Rank-biserial correlation, Cliff's delta, standardized paired mean
    difference.
:mod:`stats.intervals`
    Bootstrap confidence intervals for setting-level paired differences,
    BCa by default.
:mod:`stats.equivalence`
    TOST (equivalence margin ``delta`` is a required parameter, never defaulted)
    and a Bayesian correlated t-test skeleton with the Nadeau--Bengio correction.
:mod:`stats.cd_diagram`
    Demsar critical-difference diagrams.
"""

from . import (  # noqa: F401
    cd_diagram,
    effect_size,
    equivalence,
    intervals,
    long_table,
    omnibus,
    posthoc,
)

__all__ = [
    "long_table",
    "omnibus",
    "posthoc",
    "effect_size",
    "intervals",
    "equivalence",
    "cd_diagram",
]
