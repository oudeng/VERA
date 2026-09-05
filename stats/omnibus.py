"""Omnibus tests across multiple methods: Friedman + Iman--Davenport.

R0 never ran an omnibus test.  Its only inferential procedure was a family of
one-vs-one Wilcoxon tests (``ext2/scripts/exp5_significance_tests.py``), which
is exactly the design reviewer R1-5 objects to: pairwise-only comparisons with
no global test and no effect sizes.

Demsar (2006) prescribes Friedman on the average ranks, followed by the
Iman--Davenport correction because Friedman's chi-square is known to be
conservative for small ``k`` and ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "FriedmanResult",
    "rank_matrix",
    "average_ranks",
    "friedman_test",
    "iman_davenport",
    "friedman_from_matrix",
]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def rank_matrix(matrix: pd.DataFrame, *, higher_is_better: bool) -> pd.DataFrame:
    """Rank methods within each block (row).  Rank 1 = best.

    Ties receive the average rank, the convention Demsar assumes.
    """
    values = matrix.to_numpy(dtype=float)
    if higher_is_better:
        values = -values  # rank ascending => best first
    ranks = np.apply_along_axis(lambda r: stats.rankdata(r, method="average"), 1, values)
    return pd.DataFrame(ranks, index=matrix.index, columns=matrix.columns)


def average_ranks(matrix: pd.DataFrame, *, higher_is_better: bool) -> pd.Series:
    """Average rank per method across blocks (lower = better)."""
    return rank_matrix(matrix, higher_is_better=higher_is_better).mean(axis=0)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@dataclass
class FriedmanResult:
    metric: str
    n_blocks: int
    k_methods: int
    chi2: float
    chi2_df: int
    chi2_p: float
    F_iman_davenport: float
    F_df1: int
    F_df2: int
    F_p: float
    avg_ranks: Dict[str, float]
    higher_is_better: bool

    def to_row(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("avg_ranks")
        d["best_method"] = min(self.avg_ranks, key=self.avg_ranks.get)
        d["best_avg_rank"] = min(self.avg_ranks.values())
        d["worst_method"] = max(self.avg_ranks, key=self.avg_ranks.get)
        d["worst_avg_rank"] = max(self.avg_ranks.values())
        return d


def friedman_test(matrix: pd.DataFrame, *, higher_is_better: bool) -> tuple[float, int, float, pd.Series]:
    """Friedman chi-square from a ``block x method`` matrix.

    Returns ``(chi2, df, p, avg_ranks)``.  The statistic is computed from the
    average ranks (tie-corrected form) rather than delegating to
    ``scipy.stats.friedmanchisquare``, so that the same ranks feed the
    Iman--Davenport correction and the CD diagram.  The two agree to floating
    point for tie-free data; see ``tests/test_stats.py``.
    """
    if matrix.isna().to_numpy().any():
        raise ValueError("Friedman requires a complete block design; matrix has NaNs")
    n, k = matrix.shape
    if n < 2 or k < 2:
        raise ValueError(f"need >=2 blocks and >=2 methods, got n={n}, k={k}")

    ranks = rank_matrix(matrix, higher_is_better=higher_is_better)
    r_bar = ranks.mean(axis=0)

    # Conover's tie-corrected form:
    #   T1 = (k-1) * (sum_j R_j^2 - n*C1) / (A1 - C1)
    # with C1 = n*k*(k+1)^2/4, A1 = sum_ij rank_ij^2, R_j = sum_i rank_ij.
    # Reduces to the textbook 12/(nk(k+1)) * sum R_j^2 - 3n(k+1) when tie-free.
    rank_vals = ranks.to_numpy(dtype=float)
    a1 = float((rank_vals ** 2).sum())
    c1 = n * k * (k + 1) ** 2 / 4.0
    sum_rj_sq = float((rank_vals.sum(axis=0) ** 2).sum())
    denom = a1 - c1
    if denom <= 0:  # all methods tied in every block
        chi2 = 0.0
    else:
        chi2 = (k - 1) * (sum_rj_sq - n * c1) / denom
    df = k - 1
    p = float(stats.chi2.sf(chi2, df))
    return float(chi2), int(df), p, r_bar


def iman_davenport(chi2: float, n_blocks: int, k_methods: int) -> tuple[float, int, int, float]:
    """Iman--Davenport F correction of a Friedman chi-square.

    ``F = (n-1) * chi2 / (n*(k-1) - chi2)`` with ``(k-1, (k-1)(n-1))`` df.
    """
    n, k = int(n_blocks), int(k_methods)
    denom = n * (k - 1) - chi2
    if denom <= 0:
        F = float("inf")
        p = 0.0
    else:
        F = (n - 1) * chi2 / denom
        p = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1)))
    return float(F), int(k - 1), int((k - 1) * (n - 1)), p


def friedman_from_matrix(
    matrix: pd.DataFrame, *, metric: str, higher_is_better: bool
) -> FriedmanResult:
    """Convenience wrapper returning a fully populated :class:`FriedmanResult`."""
    chi2, df, p, r_bar = friedman_test(matrix, higher_is_better=higher_is_better)
    F, df1, df2, pf = iman_davenport(chi2, matrix.shape[0], matrix.shape[1])
    return FriedmanResult(
        metric=metric,
        n_blocks=int(matrix.shape[0]),
        k_methods=int(matrix.shape[1]),
        chi2=chi2,
        chi2_df=df,
        chi2_p=p,
        F_iman_davenport=F,
        F_df1=df1,
        F_df2=df2,
        F_p=pf,
        avg_ranks={str(k_): float(v) for k_, v in r_bar.items()},
        higher_is_better=bool(higher_is_better),
    )
