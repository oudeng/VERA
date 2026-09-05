from __future__ import annotations

"""Shared baseline helpers.

Two families of function live here:

* the **R1 (de-leaked)** helpers, which only ever see the incomplete table and
  an :class:`~baselines.schema.ObservedStats` fitted from it;
* the **legacy (oracle)** helpers, which are byte-for-byte ports of
  ``project_sni_R0/sni/baselines/utils.py`` and exist solely so that the
  before/after impact study in ``results/T1.4_deleak`` can run the R0 behavior
  unchanged. They are never reachable from the default code path -- only from
  ``registry.build_baseline_imputer(..., legacy_oracle=True)``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import DataSchema, ObservedStats

__all__ = [
    "fallback_fillna",
    "apply_observed_categories",
    "set_categories_from_complete",
    "fallback_fillna_oracle",
]


# ---------------------------------------------------------------------------
# R1: de-leaked helpers
# ---------------------------------------------------------------------------

def fallback_fillna(
    X_imputed: pd.DataFrame,
    stats: ObservedStats,
    categorical_vars: List[str],
    continuous_vars: List[str],
) -> pd.DataFrame:
    """Fill residual NaNs using statistics fitted on the *incomplete* table.

    R1 replacement for ``project_sni_R0/sni/baselines/utils.py:53-87``, which
    filled from ``X_complete``'s mean/mode -- i.e. the ground truth. Every
    baseline wrapper in R0 called it (``registry.py:66,82,117,154,196,252,283,331``),
    so this single function leaked the oracle into all eight methods regardless
    of what the underlying implementation did.

    The safety-net role is unchanged: some implementations can legitimately leave
    a cell NaN (e.g. KNN when a column has no valid donor), and the evaluator
    requires finite values at every evaluation position.
    """
    out = X_imputed.copy()

    for col in continuous_vars:
        if col not in out.columns:
            continue
        if out[col].isna().any():
            fill = stats.cont_mean.get(col, 0.0)
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill)

    for col in categorical_vars:
        if col not in out.columns:
            continue
        if out[col].isna().any():
            fill = stats.cat_mode.get(col, None)
            if fill is None or (isinstance(fill, float) and np.isnan(fill)):
                continue
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                if fill not in list(out[col].cat.categories):
                    out[col] = out[col].cat.add_categories([fill])
            out[col] = out[col].fillna(fill)

    return out


def apply_observed_categories(
    X_missing: pd.DataFrame,
    stats: ObservedStats,
) -> pd.DataFrame:
    """R1 replacement for ``set_categories_from_complete``.

    Returns a single frame (there is no second, complete frame to align with any
    more) whose categorical columns carry the vocabulary observed in the fitted
    table.
    """
    return stats.apply_categories(X_missing)


# ---------------------------------------------------------------------------
# Legacy (oracle) helpers -- verbatim ports of the R0 implementations.
# Reachable only via legacy_oracle=True. Kept so the impact study can measure
# the leak instead of arguing about it.
# ---------------------------------------------------------------------------

def set_categories_from_complete(
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    categorical_vars: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """LEGACY / ORACLE. Verbatim port of R0 ``baselines/utils.py:8-50``.

    Takes the category vocabulary from the ground-truth table and stamps it onto
    both frames. This is how GAIN, MIWAE, MissForest, HyperImpute, TabCSDI and
    MICE all received an oracle vocabulary in R0 even when their own code only
    looked at ``X_missing``.
    """
    Xc = X_complete.copy()
    Xm = X_missing.copy()

    for col in categorical_vars:
        if col not in Xc.columns or col not in Xm.columns:
            continue

        if isinstance(Xc[col].dtype, pd.CategoricalDtype):
            cats = list(Xc[col].cat.categories)
        else:
            cats = pd.Series(Xc[col].dropna().unique()).tolist()
            try:
                cats = sorted(cats)
            except Exception:
                pass

        Xc[col] = pd.Categorical(Xc[col], categories=cats)
        Xm[col] = pd.Categorical(Xm[col], categories=cats)

    return Xc, Xm


def fallback_fillna_oracle(
    X_imputed: pd.DataFrame,
    X_complete: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
) -> pd.DataFrame:
    """LEGACY / ORACLE. Verbatim port of R0 ``baselines/utils.py:53-87``."""
    out = X_imputed.copy()

    for col in continuous_vars:
        if col not in out.columns or col not in X_complete.columns:
            continue
        if out[col].isna().any():
            mean_val = pd.to_numeric(X_complete[col], errors="coerce").mean(skipna=True)
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(mean_val)

    for col in categorical_vars:
        if col not in out.columns or col not in X_complete.columns:
            continue
        if out[col].isna().any():
            mode_series = X_complete[col].mode(dropna=True)
            if len(mode_series) > 0:
                mode_val = mode_series.iloc[0]
                if isinstance(out[col].dtype, pd.CategoricalDtype):
                    if mode_val not in list(out[col].cat.categories):
                        out[col] = out[col].cat.add_categories([mode_val])
                out[col] = out[col].fillna(mode_val)

    return out
