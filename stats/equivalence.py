"""Equivalence testing: TOST, and a Bayesian correlated t-test skeleton.

Why this module exists
----------------------
Both reviewers make the same point.  R1-5: "a non-significant result against
HyperImpute does not demonstrate equivalence."  R2-6a: "twelve paired settings
are underpowered; failing to reject is not the same as equivalence."  They are
right, and the fix is to state an equivalence margin and test *for* equivalence
rather than reading a large p-value as agreement.

The equivalence margin
----------------------
``delta`` is a **required parameter with no default anywhere in this module**.
Choosing it is a methodological decision reserved for the first author: it
asserts how large an NRMSE (or R^2, or Macro-F1) difference would still be
practically irrelevant for an auditing workflow.  Hardcoding a value here would
quietly make that decision on their behalf and would be indefensible under
review.  :func:`tost_sensitivity` therefore reports the whole delta curve and
lets the reader see at which margin the conclusion flips.

Contents
--------
:func:`tost_paired`
    Two one-sided t-tests on paired differences.  Equivalence is declared when
    *both* one-sided tests reject, equivalently when the ``(1-2*alpha)``
    interval lies entirely inside ``(-delta, +delta)``.
:func:`tost_sensitivity`
    The same test swept over candidate margins -- a sensitivity table, not a
    verdict.
:func:`tost_wilcoxon_paired`
    Non-parametric TOST, for when 12 settings make normality unappealing.
:func:`nadeau_bengio_variance_factor` / :func:`bayesian_correlated_ttest`
    Skeleton of the Bayesian correlated t-test (Nadeau & Bengio 2003; Corani &
    Benavoli 2015).  Marked SKELETON: the correlation term ``rho`` is only
    identified for resampling designs where the train/test split sizes are
    known.  Across *settings* (dataset x mechanism) the pairs are not resamples
    of one population and ``rho`` has no design-implied value, so it must be
    supplied explicitly.  Do not report this as a headline number before P4.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TOSTResult",
    "tost_paired",
    "tost_sensitivity",
    "tost_wilcoxon_paired",
    "nadeau_bengio_variance_factor",
    "BayesianCorrelatedTResult",
    "bayesian_correlated_ttest",
]


class MarginNotSpecified(ValueError):
    """Raised when an equivalence margin was not supplied."""


def _require_delta(delta: Optional[float]) -> float:
    if delta is None:
        raise MarginNotSpecified(
            "the equivalence margin `delta` has no default: it is a methodological "
            "decision reserved for the first author. Supply an explicit value, or "
            "use tost_sensitivity() to report the whole delta curve."
        )
    d = float(delta)
    if not np.isfinite(d) or d <= 0:
        raise ValueError(f"delta must be a positive finite number, got {delta!r}")
    return d


# --------------------------------------------------------------------------- #
# TOST
# --------------------------------------------------------------------------- #


@dataclass
class TOSTResult:
    n: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    delta: float
    alpha: float
    t_lower: float
    p_lower: float
    t_upper: float
    p_upper: float
    p_tost: float
    ci_low: float
    ci_high: float
    ci_level: float
    equivalent: bool
    verdict: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def tost_paired(
    diffs: Sequence[float],
    delta: Optional[float] = None,
    *,
    alpha: float = 0.05,
) -> TOSTResult:
    """Two one-sided tests for equivalence on paired differences.

    Parameters
    ----------
    diffs
        Paired differences, already signed so that positive favors the
        reference method (see :func:`stats.posthoc.wilcoxon_holm`).
    delta
        **Required.** The equivalence margin, on the metric's own scale.  The
        null hypotheses are ``mean <= -delta`` and ``mean >= +delta``.
    alpha
        One-sided level for each test.  The corresponding interval is
        ``1 - 2*alpha`` (90% for the conventional ``alpha=0.05``).
    """
    d_margin = _require_delta(delta)
    x = np.asarray(diffs, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        raise ValueError(f"TOST needs >=2 paired differences, got {n}")

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    df = n - 1

    if se == 0.0:
        # Degenerate: all differences identical.
        equivalent = abs(mean) < d_margin
        return TOSTResult(
            n=n, mean_diff=mean, sd_diff=sd, se_diff=se, delta=d_margin, alpha=alpha,
            t_lower=float("inf") if equivalent else float("-inf"), p_lower=0.0 if equivalent else 1.0,
            t_upper=float("-inf") if equivalent else float("inf"), p_upper=0.0 if equivalent else 1.0,
            p_tost=0.0 if equivalent else 1.0,
            ci_low=mean, ci_high=mean, ci_level=1 - 2 * alpha,
            equivalent=bool(equivalent),
            verdict="equivalent (degenerate: zero variance)" if equivalent
            else "not equivalent (degenerate: zero variance)",
        )

    t_lower = (mean + d_margin) / se     # H0: mean <= -delta
    p_lower = float(stats.t.sf(t_lower, df))
    t_upper = (mean - d_margin) / se     # H0: mean >= +delta
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)

    tcrit = float(stats.t.ppf(1.0 - alpha, df))
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se
    equivalent = bool(p_tost < alpha)

    if equivalent:
        verdict = "equivalent at this margin"
    elif ci_low > d_margin or ci_high < -d_margin:
        verdict = "different (interval outside the margin)"
    else:
        verdict = "inconclusive (interval straddles the margin)"

    return TOSTResult(
        n=n, mean_diff=mean, sd_diff=sd, se_diff=se, delta=d_margin, alpha=alpha,
        t_lower=float(t_lower), p_lower=p_lower, t_upper=float(t_upper), p_upper=p_upper,
        p_tost=float(p_tost), ci_low=float(ci_low), ci_high=float(ci_high),
        ci_level=1 - 2 * alpha, equivalent=equivalent, verdict=verdict,
    )


def tost_sensitivity(
    diffs: Sequence[float],
    deltas: Iterable[float],
    *,
    alpha: float = 0.05,
    label: str = "",
) -> pd.DataFrame:
    """Sweep TOST across candidate margins.

    Deliberately returns the whole curve.  The point at which ``equivalent``
    flips from ``False`` to ``True`` is the smallest margin the data can
    support; reporting that boundary is honest, whereas picking one delta and
    reporting only its verdict is not.
    """
    rows = []
    for d in deltas:
        r = tost_paired(diffs, d, alpha=alpha)
        row = r.to_dict()
        if label:
            row = {"comparison": label, **row}
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("delta").reset_index(drop=True)
    return out


def tost_wilcoxon_paired(
    diffs: Sequence[float],
    delta: Optional[float] = None,
    *,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Non-parametric TOST: two one-sided Wilcoxon signed-rank tests.

    Shifts the differences by ``+/- delta`` and tests each side.  Preferable to
    the t-based TOST when n is small and the differences are visibly skewed,
    which is the case for the 12 published settings.
    """
    d_margin = _require_delta(delta)
    x = np.asarray(diffs, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        raise ValueError(f"Wilcoxon TOST needs >=3 paired differences, got {n}")

    def _one_sided(shifted: np.ndarray, alternative: str) -> float:
        if np.allclose(shifted, 0.0):
            return 1.0
        try:
            _, p = stats.wilcoxon(shifted, alternative=alternative, zero_method="wilcox")
            return float(p)
        except Exception:  # pragma: no cover
            return float("nan")

    p_lower = _one_sided(x + d_margin, "greater")   # H0: median <= -delta
    p_upper = _one_sided(x - d_margin, "less")      # H0: median >= +delta
    p_tost = max(p_lower, p_upper)
    return {
        "n": int(n),
        "median_diff": float(np.median(x)),
        "delta": d_margin,
        "alpha": float(alpha),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": float(p_tost),
        "equivalent": bool(np.isfinite(p_tost) and p_tost < alpha),
        "test": "Wilcoxon TOST (two one-sided signed-rank tests)",
    }


# --------------------------------------------------------------------------- #
# Bayesian correlated t-test -- SKELETON
# --------------------------------------------------------------------------- #


def nadeau_bengio_variance_factor(n: int, n_train: int, n_test: int) -> float:
    """Nadeau--Bengio corrected variance factor ``1/n + n_test/n_train``.

    The naive paired-t variance ``sigma^2/n`` is anti-conservative for resampled
    train/test splits because the folds overlap.  Nadeau & Bengio (2003) replace
    ``1/n`` with ``1/n + n_test/n_train``.  Corani & Benavoli (2015) write the
    same correction as ``1/n + rho/(1-rho)`` with ``rho = n_test/(n_train+n_test)``.
    """
    if n < 1 or n_train <= 0 or n_test <= 0:
        raise ValueError("n, n_train and n_test must be positive")
    return float(1.0 / n + float(n_test) / float(n_train))


@dataclass
class BayesianCorrelatedTResult:
    n: int
    mean_diff: float
    sd_diff: float
    rope: tuple
    rho: float
    variance_factor: float
    p_left: float
    p_rope: float
    p_right: float
    decision: str
    status: str = "SKELETON"
    caveat: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bayesian_correlated_ttest(
    diffs: Sequence[float],
    rope: Optional[tuple] = None,
    *,
    rho: Optional[float] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
) -> BayesianCorrelatedTResult:
    """SKELETON: Bayesian correlated t-test with the Nadeau--Bengio correction.

    Implements the standard closed form (Corani & Benavoli 2015): with a matched
    (improper) prior the posterior of the mean difference is Student-t with
    ``df = n-1``, location ``mean(diff)`` and scale
    ``sd(diff) * sqrt(1/n + rho/(1-rho))``.  Integrating that posterior over the
    ROPE gives ``P(left) / P(rope) / P(right)``.

    Why SKELETON
    ------------
    1. ``rope`` is the Bayesian counterpart of the TOST margin ``delta`` and is
       likewise **required, with no default**.
    2. ``rho`` is only design-identified for cross-validation-style resampling,
       where ``rho = n_test/(n_train+n_test)``.  The 12 units here are distinct
       *settings*, not resamples of one population, so no design value exists.
       Pass ``rho`` explicitly (with a written justification), or pass
       ``n_train``/``n_test`` when the pairing really is a resampling design --
       which is the case for the per-seed analysis inside a single setting, not
       for the across-settings analysis.
    3. Not to be quoted as a headline result before P4 fixes both choices.
    """
    if rope is None:
        raise MarginNotSpecified(
            "the ROPE has no default: like the TOST margin it is a methodological "
            "decision reserved for the first author."
        )
    lo, hi = float(rope[0]), float(rope[1])
    if not lo < hi:
        raise ValueError(f"rope must be (low, high) with low < high, got {rope!r}")

    x = np.asarray(diffs, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        raise ValueError(f"need >=2 paired differences, got {n}")

    if rho is None:
        if n_train is None or n_test is None:
            raise ValueError(
                "supply either rho, or n_train and n_test so that "
                "rho = n_test/(n_train+n_test) is design-identified"
            )
        rho = float(n_test) / float(n_train + n_test)
    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must lie in [0, 1), got {rho}")

    var_factor = 1.0 / n + rho / (1.0 - rho) if rho > 0 else 1.0 / n
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    scale = sd * np.sqrt(var_factor)
    df = n - 1

    if scale <= 0:
        p_left = float(mean < lo)
        p_rope = float(lo <= mean <= hi)
        p_right = float(mean > hi)
    else:
        cdf_lo = float(stats.t.cdf((lo - mean) / scale, df))
        cdf_hi = float(stats.t.cdf((hi - mean) / scale, df))
        p_left = cdf_lo
        p_rope = cdf_hi - cdf_lo
        p_right = 1.0 - cdf_hi

    probs = {"left": p_left, "rope": p_rope, "right": p_right}
    decision = max(probs, key=probs.get)

    return BayesianCorrelatedTResult(
        n=int(n),
        mean_diff=mean,
        sd_diff=sd,
        rope=(lo, hi),
        rho=rho,
        variance_factor=float(var_factor),
        p_left=float(p_left),
        p_rope=float(p_rope),
        p_right=float(p_right),
        decision=decision,
        status="SKELETON",
        caveat=(
            "rho is not design-identified for across-setting pairing; ROPE and rho "
            "must both be fixed by the first author before this is reported."
        ),
    )
