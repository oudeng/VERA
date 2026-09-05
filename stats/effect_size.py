"""Effect sizes for paired method comparisons (R1-5).

R0 has **no** effect-size implementation anywhere in the repository.  The
abandoned attempt is still visible in ``scripts/06_gen_supp_tables.py:293-297``:

    # Compute effect size r = Z / sqrt(N)
    # ... We don't have Z directly; let's compute from W and n
    # Actually, let's just report p-value + mean_diff as the key info

The give-up was unnecessary.  ``results_all/ext2/significance/
wilcoxon_across_settings.csv`` already stores ``W_statistic`` and
``n_settings = 12``, and the matched-pairs rank-biserial correlation is a closed
form in exactly those two numbers::

    |r_rb| = 1 - 2W / (n(n+1)/2)

because SciPy's two-sided ``wilcoxon`` returns ``W = min(W+, W-)`` and
``W+ + W- = n(n+1)/2``.  The sign is recovered from the stored ``mean_diff``.
So every number in Table S3 can be upgraded to carry an effect size with
**zero recomputation**; :func:`rank_biserial_from_w` does precisely that.

Three families are provided:

* :func:`rank_biserial_from_w` / :func:`rank_biserial_from_diffs` -- the
  non-parametric effect size that matches the Wilcoxon test actually reported.
* :func:`cliffs_delta` (independent samples) and :func:`cliffs_delta_paired`
  (dominance of positive over negative differences) -- ordinal, assumption-free.
* :func:`standardized_paired_mean_difference` -- Cohen's ``d_z`` with the
  Hedges small-sample correction, for readers who expect a ``d``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MAGNITUDE_THRESHOLDS",
    "EffectSize",
    "rank_biserial_from_w",
    "rank_biserial_from_diffs",
    "cliffs_delta",
    "cliffs_delta_paired",
    "standardized_paired_mean_difference",
    "interpret_magnitude",
    "effect_sizes_for_diffs",
    "augment_r0_wilcoxon_table",
    "R0_EFFECTIVE_N",
]

#: Romano et al. (2006) thresholds, the convention for Cliff's delta and the
#: matched-pairs rank-biserial correlation.
MAGNITUDE_THRESHOLDS: Dict[str, float] = {
    "negligible": 0.147,
    "small": 0.330,
    "medium": 0.474,
}


def interpret_magnitude(value: float) -> str:
    """Map |effect| to negligible / small / medium / large."""
    if value is None or not np.isfinite(value):
        return "undefined"
    a = abs(float(value))
    if a < MAGNITUDE_THRESHOLDS["negligible"]:
        return "negligible"
    if a < MAGNITUDE_THRESHOLDS["small"]:
        return "small"
    if a < MAGNITUDE_THRESHOLDS["medium"]:
        return "medium"
    return "large"


@dataclass
class EffectSize:
    n: int
    rank_biserial: float
    rank_biserial_magnitude: str
    cliffs_delta_paired: float
    cliffs_delta_magnitude: str
    cohens_dz: float
    hedges_gz: float
    mean_diff: float
    median_diff: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Rank-biserial correlation
# --------------------------------------------------------------------------- #


def rank_biserial_from_w(
    w_statistic: float, n: int, *, sign: Optional[float] = None
) -> float:
    """Matched-pairs rank-biserial correlation from a stored Wilcoxon ``W``.

    Parameters
    ----------
    w_statistic
        SciPy's two-sided ``wilcoxon`` statistic, i.e. ``min(W+, W-)``.
    n
        Number of pairs used by the test (non-zero differences under
        ``zero_method='wilcox'``).
    sign
        Any quantity whose sign indicates the direction of the effect --
        typically the stored ``mean_diff``.  ``None`` returns the unsigned
        magnitude.

    Notes
    -----
    ``W+ + W- = n(n+1)/2 =: T`` and ``r_rb = (W+ - W-)/T``.  With ``W = min``,
    ``|r_rb| = (T - 2W)/T = 1 - 2W/T``.  Because only the minimum is stored,
    the sign has to come from elsewhere -- hence ``sign``.
    """
    n = int(n)
    if n < 1:
        return float("nan")
    total = n * (n + 1) / 2.0
    if total <= 0 or not np.isfinite(w_statistic):
        return float("nan")
    magnitude = 1.0 - 2.0 * float(w_statistic) / total
    magnitude = float(np.clip(magnitude, -1.0, 1.0))
    if sign is None:
        return magnitude
    s = np.sign(float(sign))
    if s == 0:
        return 0.0
    return float(s * abs(magnitude))


def rank_biserial_from_diffs(diffs: Sequence[float], *, zero_method: str = "wilcox") -> float:
    """Signed matched-pairs rank-biserial computed directly from differences.

    Used to validate :func:`rank_biserial_from_w`; the two agree to floating
    point on the R0 table (see ``tests/test_stats.py``).
    """
    from scipy.stats import rankdata

    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if zero_method == "wilcox":
        d = d[d != 0]
    if d.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(d), method="average")
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    total = w_pos + w_neg
    if total <= 0:
        return 0.0
    return float((w_pos - w_neg) / total)


# --------------------------------------------------------------------------- #
# Cliff's delta
# --------------------------------------------------------------------------- #


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta for two independent samples: ``P(x>y) - P(x<y)``."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / (a.size * b.size))


def cliffs_delta_paired(diffs: Sequence[float]) -> float:
    """Dominance statistic on paired differences: ``P(d>0) - P(d<0)``.

    This is the ordinal effect size that pairs with the sign test; it is
    reported alongside the rank-biserial correlation because it is insensitive
    to the magnitude of individual differences and therefore robust to the one
    or two settings where a baseline collapses (e.g. GAIN's ``R2 = -1.13`` on
    AutoMPG).
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    return float((np.sign(d).sum()) / d.size)


# --------------------------------------------------------------------------- #
# Standardized mean difference
# --------------------------------------------------------------------------- #


def standardized_paired_mean_difference(
    diffs: Sequence[float], *, hedges: bool = True
) -> tuple[float, float]:
    """Cohen's ``d_z`` for paired data, and its Hedges-corrected ``g_z``.

    ``d_z = mean(diff) / sd(diff)`` with ``sd`` the sample standard deviation
    (ddof=1).  The Hedges factor ``J = 1 - 3/(4*df - 1)`` removes the small-sample
    upward bias; with ``n = 12`` it is about 0.93, which is not negligible.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        dz = 0.0 if np.allclose(d, 0.0) else float(np.inf) * np.sign(float(np.mean(d)))
    else:
        dz = float(np.mean(d) / sd)
    if not hedges or not np.isfinite(dz):
        return dz, dz
    df = n - 1
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return dz, float(J * dz)


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def effect_sizes_for_diffs(diffs: Sequence[float]) -> EffectSize:
    """All three effect-size families for one difference vector."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    r_rb = rank_biserial_from_diffs(d)
    delta = cliffs_delta_paired(d)
    dz, gz = standardized_paired_mean_difference(d)
    return EffectSize(
        n=int(d.size),
        rank_biserial=float(r_rb),
        rank_biserial_magnitude=interpret_magnitude(r_rb),
        cliffs_delta_paired=float(delta),
        cliffs_delta_magnitude=interpret_magnitude(delta),
        cohens_dz=float(dz),
        hedges_gz=float(gz),
        mean_diff=float(np.mean(d)) if d.size else float("nan"),
        median_diff=float(np.median(d)) if d.size else float("nan"),
    )


#: The ``n_settings`` column of R0's stored Wilcoxon table is the number of
#: merged settings, **not** the number of pairs the test actually used.  For the
#: categorical metrics Concrete contributes a NaN (it has no categorical
#: columns), ``_wilcoxon_safe`` drops it, and the effective n is 10 -- while the
#: CSV still says 12.  Using 12 would inflate ``T = n(n+1)/2`` from 55 to 78 and
#: silently shrink every categorical effect size.
R0_EFFECTIVE_N: Dict[str, int] = {
    "NRMSE": 12,
    "R2": 12,
    "Spearman_rho": 12,
    "Macro_F1": 10,
    "Accuracy": 10,
    "Cohens_kappa": 10,
}


def augment_r0_wilcoxon_table(
    path_or_frame,
    *,
    effective_n: Optional[Dict[str, int]] = None,
    sign_override: Optional[Dict[tuple, float]] = None,
) -> pd.DataFrame:
    """Add rank-biserial effect sizes to R0's stored Wilcoxon table.

    Zero recomputation: ``W_statistic`` and the pair count are all that is
    needed, and both are already in
    ``results_all/ext2/significance/wilcoxon_across_settings.csv``.

    Two corrections are applied, and both are recorded in output columns so the
    table is self-documenting:

    ``n_effective``
        Defaults to :data:`R0_EFFECTIVE_N`, which overrides the stored
        ``n_settings`` for the categorical metrics (see the constant's docs).
        Pass ``effective_n={}`` to trust the stored value instead.
    ``sign_source``
        The sign of ``r_rb`` is taken from ``mean_diff``, the only direction
        carrier in the CSV.  For a skewed difference vector the mean can point
        the opposite way to the signed-rank sum; ``sign_override`` accepts a
        ``{(metric, comparison): value}`` map to supply the correct direction
        from a recomputed difference vector.  Rows using an override are marked.
    """
    df = (
        path_or_frame.copy()
        if isinstance(path_or_frame, pd.DataFrame)
        else pd.read_csv(path_or_frame)
    )
    required = {"W_statistic", "n_settings", "mean_diff", "metric", "comparison"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"table lacks required columns {sorted(missing)}")

    eff = R0_EFFECTIVE_N if effective_n is None else effective_n
    sign_override = sign_override or {}

    n_eff, r_rb, sign_src = [], [], []
    for _, row in df.iterrows():
        metric = str(row["metric"])
        n = int(eff.get(metric, row["n_settings"]))
        key = (metric, str(row["comparison"]))
        if key in sign_override:
            s = float(sign_override[key])
            src = "override"
        else:
            s = float(row["mean_diff"])
            src = "mean_diff"
        n_eff.append(n)
        sign_src.append(src)
        r_rb.append(rank_biserial_from_w(row["W_statistic"], n, sign=s))

    df["n_effective"] = n_eff
    df["n_effective_differs_from_stored"] = df["n_effective"] != df["n_settings"]
    df["sign_source"] = sign_src
    df["rank_biserial"] = r_rb
    df["rank_biserial_magnitude"] = df["rank_biserial"].map(interpret_magnitude)
    return df
