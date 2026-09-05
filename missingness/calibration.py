"""Rate calibration in logit space (P0 findings B39 and the MAR mis-calibration).

Two defects of R0 are fixed here.

1. **Mean-linear rescaling distorts the sigmoid.**
   ``missing_data_generator._mar_propensity`` (``missing_data_generator.py:460-465``)
   built ``p = sigmoid(z)`` and then hit it with a multiplicative constant::

       mean_p  = float(np.mean(p_row))
       p_row   = p_row * (rate / mean_p)
       p_row   = np.clip(p_row, 0.0, 1.0)

   That is not a logistic model any more. It changes the *shape* of the
   propensity, not just its level: at ``rate = 0.5`` the factor is ~1.0 and the
   curve is untouched, at ``rate = 0.1`` the factor is ~0.2 and the curve is
   flattened towards linearity, and the ``clip`` at 1.0 silently truncates the
   upper tail whenever ``rate`` is large. So the *effective* strength of the MAR
   dependence varied with the target rate — the 10%/30%/50% masks of the same
   dataset are not the same mechanism at three levels.

   :func:`solve_intercept` instead solves ``mean(sigmoid(eta + b)) = rate`` for
   the intercept ``b``. The slope, and hence the mechanism, is invariant to the
   target rate. This is the method already used (correctly) by R0's synthetic
   generator, ``scripts/synth_generate_s5.py:76-87``, ported and hardened here.

2. **Per-column rates were never controlled.**
   R0 calibrated the *table* total over eligible cells
   (``missing_data_generator.calibrate_mask_to_rate``,
   ``missing_data_generator.py:324-411``). Individual columns were free to drift,
   and did: at a 30% target, MIMIC ``ALARM`` came out at 37.8%, eICU
   ``vasopressor_use_std`` at 43.4%, ``mechanical_ventilation_std`` at 39.7%,
   ``composite_risk_score`` at 34.2% and NHANES ``gender_std`` at 34.0%
   (finding B39). :func:`calibrate_column_to_count` pins each column
   individually.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#: Ceiling on |logit| used when a probability has to be turned back into a logit.
_LOGIT_CLIP = 1e-12


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function.

    R0 ``scripts/synth_generate_s5.py:68-69`` used the naive
    ``1/(1+exp(-x))``, which overflows in ``exp`` for ``x <~ -710`` and emits a
    RuntimeWarning long before that. The bisection below evaluates the sigmoid at
    intercepts as extreme as +/-60, so the stable branch matters.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def solve_intercept(
    eta: np.ndarray,
    target_rate: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
    bracket: float = 60.0,
) -> float:
    """Return ``b`` with ``mean(sigmoid(eta + b)) == target_rate``.

    ``eta`` is the linear predictor *without* intercept, i.e. already
    ``sum_k coef_k * z_k`` for MAR or ``a * z_self`` for MNAR.

    ``mean(sigmoid(eta + b))`` is continuous and strictly increasing in ``b``
    (it is a mean of strictly increasing functions), so plain bisection is
    exact to machine precision. R0's synthetic version
    (``scripts/synth_generate_s5.py:76-87``) hard-coded a ``[-12, 12]`` bracket
    and 60 iterations with no verification that the root was inside; with a
    strong slope and a 10% target the root can sit outside that bracket and the
    function then silently returns the boundary. Here the bracket is widened
    until it provably contains the root, and the achieved mean is returned to the
    caller for recording.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.size == 0:
        raise ValueError("solve_intercept: empty linear predictor")
    r = float(target_rate)
    if not (0.0 < r < 1.0):
        raise ValueError(f"target_rate must be in (0,1), got {target_rate!r}")

    lo, hi = -abs(bracket), abs(bracket)
    # Widen until the root is bracketed (only needed for extreme eta / rates).
    for _ in range(12):
        if sigmoid(eta + lo).mean() <= r <= sigmoid(eta + hi).mean():
            break
        lo *= 2.0
        hi *= 2.0
    else:  # pragma: no cover - would need a pathological eta
        raise RuntimeError(
            f"solve_intercept: could not bracket target_rate={r} "
            f"(eta range [{eta.min():.3g}, {eta.max():.3g}])"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = float(sigmoid(eta + mid).mean())
        if abs(m - r) <= tol or (hi - lo) < 1e-14:
            return float(mid)
        if m > r:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def fit_logistic_propensity(
    eta: np.ndarray, target_rate: float, **kw
) -> Tuple[np.ndarray, float, float]:
    """``(p, b, expected_rate)`` for the calibrated logistic propensity."""
    b = solve_intercept(eta, target_rate, **kw)
    p = sigmoid(np.asarray(eta, dtype=float) + b)
    return p, float(b), float(p.mean())


def calibrate_column_to_count(
    mask_col: np.ndarray,
    propensity_col: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Snap one column's mask to ``target_count`` missing cells (fixes B39).

    Direction of adjustment preserves the mechanism's ordering within the
    column: cells are *added* at the highest propensities still observed and
    *removed* at the lowest propensities currently missing. Ties are broken with
    noise from ``rng`` — which, unlike R0, is this column's private stream, so
    the adjustment cannot perturb any other column.

    Rationale for calibrating at all: even a perfectly calibrated logistic gives
    a Bernoulli draw whose realised rate has standard deviation
    ``sqrt(p(1-p)/n)``. At ``n = 2052`` and ``p = 0.3`` that is 1.01 percentage
    points, i.e. the "within 1% of target" requirement would fail about a third
    of the time on sampling noise alone. Snapping the count removes that noise
    while leaving the *pattern* of which cells are missing set by the propensity.
    """
    mask_col = np.asarray(mask_col, dtype=bool).copy()
    p = np.asarray(propensity_col, dtype=float)
    n = mask_col.size
    target_count = int(np.clip(target_count, 0, n))
    cur = int(mask_col.sum())
    if cur == target_count:
        return mask_col

    jitter = rng.random(n) * 1e-9

    if cur < target_count:
        cand = np.flatnonzero(~mask_col)
        need = min(target_count - cur, cand.size)
        if need > 0:
            order = np.argsort(-(p[cand] + jitter[cand]), kind="stable")
            mask_col[cand[order[:need]]] = True
    else:
        cand = np.flatnonzero(mask_col)
        drop = min(cur - target_count, cand.size)
        if drop > 0:
            order = np.argsort(p[cand] + jitter[cand], kind="stable")
            mask_col[cand[order[:drop]]] = False
    return mask_col


def enforce_min_missing(
    mask_col: np.ndarray,
    propensity_col: np.ndarray,
    min_missing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Port of ``missing_data_generator._enforce_min_missing_per_column`` (:271-321).

    Behavior is unchanged — highest-propensity observed cells are flipped to
    missing — but the generator is this column's own ``min_per_col`` stream
    rather than the shared table generator (B45).
    """
    mask_col = np.asarray(mask_col, dtype=bool).copy()
    if min_missing <= 0:
        return mask_col
    cur = int(mask_col.sum())
    if cur >= min_missing:
        return mask_col
    cand = np.flatnonzero(~mask_col)
    if cand.size == 0:
        return mask_col
    need = min(min_missing - cur, cand.size)
    noise = rng.random(cand.size) * 1e-12
    order = np.argsort(-(np.asarray(propensity_col, dtype=float)[cand] + noise), kind="stable")
    mask_col[cand[order[:need]]] = True
    return mask_col


def standardize(values: np.ndarray, *, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """z-score with a zero-variance guard; returns ``(z, mean, std)``.

    A constant column (eICU ``hours_since_admission`` = 24, ``vasopressor_use_std``
    = 1.0; finding B35) yields ``z == 0`` rather than NaN, so a logistic
    propensity built on it degenerates gracefully to a constant equal to the
    target rate instead of producing NaNs or, as in R0, a uniform 1.5x
    over-rate (finding B46).
    """
    arr = np.asarray(values, dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= eps:
        return np.zeros_like(arr), mu, 0.0
    return (arr - mu) / sd, mu, sd
