"""Bootstrap confidence intervals for setting-level paired differences (R1-5).

Reviewer R1-5 asks for confidence intervals; R0 reports none anywhere.  The
paired unit here is a *setting* (dataset x mechanism), matching the unit used by
the Wilcoxon tests, so the CI answers the same question the p-value answers:
"how large is the SNI-minus-baseline difference across the 12 published
settings, and how precisely is it pinned down?"

BCa is the default because the paired-difference distribution over 12 settings
is small, skewed and occasionally has an outlier (GAIN collapses on AutoMPG),
which is exactly where the percentile interval is biased.  BCa needs the
jackknife acceleration to be well defined; when it is not (a degenerate or
constant sample) the implementation falls back to the percentile interval and
says so in ``method_used``, rather than silently returning something else.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "BootstrapCI",
    "bootstrap_ci",
    "paired_difference_ci",
    "median_ci",
]


@dataclass
class BootstrapCI:
    statistic: float
    lower: float
    upper: float
    alpha: float
    method_requested: str
    method_used: str
    n: int
    n_resamples: int
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    method: str = "bca",
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 20260727,
) -> BootstrapCI:
    """Bootstrap CI for a one-sample statistic.

    Parameters
    ----------
    method
        ``'bca'`` (default), ``'percentile'`` or ``'basic'``.  BCa degrades to
        percentile when the acceleration is undefined; the actual method is
        reported in ``method_used``.
    """
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    point = float(statistic(x)) if n else float("nan")

    if n < 2:
        return BootstrapCI(
            point, float("nan"), float("nan"), alpha, method, "insufficient_n",
            n, 0, "fewer than 2 observations",
        )
    if np.allclose(x, x[0]):
        return BootstrapCI(
            point, point, point, alpha, method, "degenerate_constant",
            n, 0, "all observations identical; interval collapses to a point",
        )

    requested = method.lower()
    scipy_method = {"bca": "BCa", "percentile": "percentile", "basic": "basic"}.get(requested)
    if scipy_method is None:
        raise ValueError(f"unknown bootstrap method {method!r}")

    rng = np.random.default_rng(seed)
    note = ""
    used = requested
    try:
        res = stats.bootstrap(
            (x,),
            statistic,
            vectorized=False,
            paired=False,
            confidence_level=1.0 - alpha,
            n_resamples=int(n_resamples),
            method=scipy_method,
            random_state=rng,
        )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)) and requested == "bca":
            raise FloatingPointError("BCa produced a non-finite endpoint")
    except Exception as exc:  # BCa can fail on degenerate jackknife
        if requested != "bca":
            raise
        note = f"BCa unavailable ({type(exc).__name__}: {exc}); fell back to percentile"
        used = "percentile"
        rng = np.random.default_rng(seed)
        res = stats.bootstrap(
            (x,),
            statistic,
            vectorized=False,
            paired=False,
            confidence_level=1.0 - alpha,
            n_resamples=int(n_resamples),
            method="percentile",
            random_state=rng,
        )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)

    return BootstrapCI(point, lo, hi, alpha, requested, used, n, int(n_resamples), note)


def paired_difference_ci(
    reference_values: Sequence[float],
    other_values: Sequence[float],
    *,
    higher_is_better: bool,
    method: str = "bca",
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 20260727,
) -> BootstrapCI:
    """CI for the mean paired difference, signed so that positive favors the reference.

    Identical sign convention to :func:`stats.posthoc.wilcoxon_holm`: the raw
    difference is ``reference - other``, negated when lower values are better.
    """
    a = np.asarray(reference_values, dtype=float)
    b = np.asarray(other_values, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align: {a.shape} vs {b.shape}")
    diff = a - b
    if not higher_is_better:
        diff = -diff
    return bootstrap_ci(
        diff, np.mean, method=method, alpha=alpha, n_resamples=n_resamples, seed=seed
    )


def median_ci(
    data: Sequence[float],
    *,
    method: str = "bca",
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 20260727,
) -> BootstrapCI:
    """Bootstrap CI for the median -- the location matching a Wilcoxon test."""
    return bootstrap_ci(
        data, np.median, method=method, alpha=alpha, n_resamples=n_resamples, seed=seed
    )
