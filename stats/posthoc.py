"""Post-hoc procedures: Nemenyi, and Wilcoxon signed-rank + Holm--Bonferroni.

Two deliberately independent routes are provided, because they answer different
questions and disagreeing answers are themselves informative:

**Nemenyi** operates on the *average ranks* produced by the Friedman test.  It
compares all methods against each other with a single critical difference and is
the standard companion to Friedman (Demsar 2006).  It is conservative: with
``k=9`` and ``n=12`` the critical difference is large.

**Wilcoxon + Holm** operates on the *raw paired values* against a single
reference method (SNI).  It reproduces R0's procedure
(``ext2/scripts/exp5_significance_tests.py:537-594``) so that the new pipeline
can be checked against the published numbers, but adds the effect sizes and
intervals R0 omitted (see :mod:`stats.effect_size`, :mod:`stats.intervals`).

Note on control-vs-all comparisons: when only SNI-vs-baseline matters, the
Bonferroni--Dunn test is the rank-based analogue of Holm; it is exposed as
:func:`bonferroni_dunn_critical_difference`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .long_table import METRIC_DIRECTION

__all__ = [
    "holm_bonferroni",
    "nemenyi_critical_difference",
    "bonferroni_dunn_critical_difference",
    "nemenyi_pvalues",
    "nemenyi_vs_reference",
    "wilcoxon_safe",
    "wilcoxon_holm",
]


# --------------------------------------------------------------------------- #
# Multiplicity control
# --------------------------------------------------------------------------- #


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Step-down Holm--Bonferroni adjusted p-values.

    Kept bit-compatible with R0's hand-rolled implementation
    (``exp5_significance_tests.py:471-486``) including its monotonicity pass, so
    that the R1 pipeline reproduces the published Table S3 exactly.  A unit test
    cross-checks it against ``statsmodels.stats.multitest.multipletests``.
    NaN p-values propagate as NaN instead of being silently ranked.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    finite = np.isfinite(p)
    adj = np.full(m, np.nan, dtype=float)
    if not finite.any():
        return adj.tolist()

    idx_finite = np.flatnonzero(finite)
    pf = p[idx_finite]
    mf = pf.size
    order = np.argsort(pf, kind="stable")
    tmp = np.empty(mf, dtype=float)
    for rank, i in enumerate(order):
        tmp[i] = min(pf[i] * (mf - rank), 1.0)
    for i in range(1, mf):
        tmp[order[i]] = max(tmp[order[i]], tmp[order[i - 1]])
    adj[idx_finite] = tmp
    return adj.tolist()


# --------------------------------------------------------------------------- #
# Nemenyi
# --------------------------------------------------------------------------- #


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi all-pairs test.

    ``CD = q_alpha * sqrt(k(k+1) / (6n))`` where ``q_alpha`` is the studentized
    range statistic at infinite df divided by ``sqrt(2)``.
    """
    q = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf)) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def bonferroni_dunn_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference when every method is compared to one control only.

    ``CD = z_{alpha/(k-1)} * sqrt(k(k+1)/(6n))``.  Tighter than Nemenyi because
    there are ``k-1`` rather than ``k(k-1)/2`` comparisons.
    """
    z = float(stats.norm.isf(alpha / (2.0 * (k - 1))))
    return float(z * np.sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_pvalues(avg_ranks: pd.Series, n_blocks: int) -> pd.DataFrame:
    """All-pairs Nemenyi p-values from average ranks (square matrix)."""
    names = list(avg_ranks.index)
    k = len(names)
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    out = pd.DataFrame(np.eye(k), index=names, columns=names, dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                out.iloc[i, j] = 1.0
                continue
            q = abs(float(avg_ranks.iloc[i]) - float(avg_ranks.iloc[j])) / se * np.sqrt(2.0)
            out.iloc[i, j] = float(stats.studentized_range.sf(q, k, np.inf))
    return out


def nemenyi_vs_reference(
    avg_ranks: pd.Series,
    n_blocks: int,
    reference: str,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Nemenyi comparisons of one reference method against all others."""
    if reference not in avg_ranks.index:
        raise KeyError(f"reference {reference!r} not among {list(avg_ranks.index)}")
    k = int(len(avg_ranks))
    cd = nemenyi_critical_difference(k, n_blocks, alpha)
    cd_bd = bonferroni_dunn_critical_difference(k, n_blocks, alpha)
    pmat = nemenyi_pvalues(avg_ranks, n_blocks)
    ref_rank = float(avg_ranks[reference])

    rows = []
    for m in avg_ranks.index:
        if m == reference:
            continue
        d = ref_rank - float(avg_ranks[m])
        rows.append(
            {
                "comparison": f"{reference} vs {m}",
                "reference": reference,
                "other": m,
                "avg_rank_reference": ref_rank,
                "avg_rank_other": float(avg_ranks[m]),
                # negative => reference has the *lower* (better) average rank
                "rank_diff": d,
                "reference_better": bool(d < 0),
                "abs_rank_diff": abs(d),
                "CD_nemenyi": cd,
                "CD_bonferroni_dunn": cd_bd,
                "significant_nemenyi": bool(abs(d) > cd),
                "significant_bonferroni_dunn": bool(abs(d) > cd_bd),
                "p_nemenyi": float(pmat.loc[reference, m]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Wilcoxon + Holm
# --------------------------------------------------------------------------- #


def wilcoxon_safe(diff: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank on a difference vector, robust to degenerate input.

    Behavior matches R0 (``exp5_significance_tests.py:488-503``): fewer than 3
    non-NaN differences -> ``(nan, nan)``; an all-zero vector -> ``(0.0, 1.0)``;
    otherwise ``alternative='two-sided'``, ``zero_method='wilcox'``.
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 3:
        return float("nan"), float("nan")
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except Exception:  # pragma: no cover
        return float("nan"), float("nan")


def wilcoxon_holm(
    matrix: pd.DataFrame,
    reference: str,
    *,
    metric: str,
    higher_is_better: Optional[bool] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Reference-vs-all Wilcoxon signed-rank tests with Holm correction.

    ``matrix`` is a ``setting x method`` table (see
    :func:`stats.long_table.to_setting_matrix`).

    Sign convention -- **the one R0's caption got backwards**.  The raw
    difference is ``reference - other``; when lower values are better it is
    negated, so that in the returned ``mean_diff`` a **positive value always
    means the reference method is better**, for every metric.  For NRMSE this is
    ``mean_diff = other - reference``.  The manuscript
    (``ESM_1_SNI_HISC_v5_5.tex:353``) states this correctly; the generator
    ``scripts/06_gen_supp_tables.py:355-360`` states it backwards.
    """
    if reference not in matrix.columns:
        raise KeyError(f"reference {reference!r} not in matrix columns {list(matrix.columns)}")
    if higher_is_better is None:
        hib = METRIC_DIRECTION.get(metric)
        if hib is None:
            raise ValueError(f"metric {metric!r} has no unambiguous direction; pass higher_is_better")
        higher_is_better = bool(hib)

    others = [c for c in matrix.columns if c != reference]
    rows: List[Dict[str, object]] = []
    pvals: List[float] = []

    for other in others:
        pair = matrix[[reference, other]].dropna()
        if len(pair) < 3:
            continue
        diff = pair[reference].to_numpy(dtype=float) - pair[other].to_numpy(dtype=float)
        if not higher_is_better:
            diff = -diff  # positive => reference better
        stat, p = wilcoxon_safe(diff)
        pvals.append(p)
        rows.append(
            {
                "metric": metric,
                "comparison": f"{reference} vs {other}",
                "reference": reference,
                "other": other,
                "test": "Wilcoxon signed-rank (paired across settings)",
                "n_settings": int(len(pair)),
                "mean_diff": float(np.nanmean(diff)),
                "median_diff": float(np.nanmedian(diff)),
                "W_statistic": stat,
                "p_value": p,
                "n_positive": int((diff > 0).sum()),
                "n_negative": int((diff < 0).sum()),
                "n_zero": int((diff == 0).sum()),
                "_diff": diff,
            }
        )

    if not rows:
        return pd.DataFrame()

    adj = holm_bonferroni(pvals)
    for row, a in zip(rows, adj):
        row["p_adjusted"] = float(a)
        row["significant"] = bool(np.isfinite(a) and a < alpha)
        row["direction"] = (
            "reference_better" if row["mean_diff"] > 0 else "reference_worse"
        )
    return pd.DataFrame(rows)
