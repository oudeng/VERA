"""Evaluation metrics layer (T1.7, part 1).

This module is a **thin wrapper** around :mod:`sni.metrics`, which is a
byte-identical port of the R0 implementation.  Nothing here re-derives a
metric: every number ultimately comes out of
:func:`sni.metrics.evaluate_imputation`, so the R0 definitions are preserved
exactly:

* ``NRMSE`` is normalized by the range of the **complete** column, not by the
  range of the evaluated (masked) subset -- see ``sni/metrics.py:81-84``.
* Evaluation happens on **masked cells only** -- see ``sni/metrics.py:214-262``.
* Summary aggregation is the unweighted mean over features within the
  continuous / categorical blocks -- ``sni/metrics.py:266-281``.

What this module adds is an **exclude-columns evaluation mode**: the ability to
score an already-produced imputation while dropping a declared set of problem
columns (T1.6a needs this for B34 / B35 / B41 / B42 / B47).  Excluding a column
is implemented by removing it from the ``categorical_vars`` / ``continuous_vars``
lists handed to the R0 evaluator.  Because the R0 evaluator loops over those
lists column-by-column and normalises each column by its own complete range,
dropping a column changes the summary **only** through the feature-average --
every retained column keeps exactly the value it had before.  That property is
asserted in ``tests/test_stats.py``.

Nothing in this module writes to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sni.metrics import EvaluationResult, evaluate_imputation

__all__ = [
    "CONTINUOUS_METRICS",
    "CATEGORICAL_METRICS",
    "SUMMARY_METRIC_COLUMNS",
    "EvaluationSpec",
    "ColumnExclusion",
    "resolve_variable_lists",
    "evaluate",
    "evaluate_many",
    "evaluate_run_directory",
    "summary_delta",
]

# Metric names exactly as produced by ``sni.metrics``.
CONTINUOUS_METRICS: Sequence[str] = ("NRMSE", "RMSE", "MAE", "MB", "R2", "Spearman")
CATEGORICAL_METRICS: Sequence[str] = ("Accuracy", "Macro-F1", "Cohen_kappa")

#: The nine metric columns that appear in ``agg_*/summary_all.csv``.
SUMMARY_METRIC_COLUMNS: Sequence[str] = tuple(
    [f"cont_{m}" for m in CONTINUOUS_METRICS] + [f"cat_{m}" for m in CATEGORICAL_METRICS]
)


# --------------------------------------------------------------------------- #
# Specification objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvaluationSpec:
    """Declares which columns take part in an evaluation.

    Parameters
    ----------
    categorical_vars, continuous_vars
        The full column roles, normally read from ``configs/datasets.yaml``.
    exclude_cols
        Columns to drop from *this* evaluation.  They are still allowed to be
        present in the frames -- they are simply never scored.  This is the
        exclude-columns mode required by T1.6a.
    name
        Human-readable label carried into result tables (e.g. ``"drop_B35"``).
    """

    categorical_vars: Sequence[str]
    continuous_vars: Sequence[str]
    exclude_cols: Sequence[str] = field(default_factory=tuple)
    name: str = "full"

    def resolved(self) -> "EvaluationSpec":
        cat, cont = resolve_variable_lists(
            self.categorical_vars, self.continuous_vars, self.exclude_cols
        )
        return EvaluationSpec(
            categorical_vars=cat,
            continuous_vars=cont,
            exclude_cols=(),
            name=self.name,
        )


@dataclass(frozen=True)
class ColumnExclusion:
    """A named exclusion rule, so that reports can cite the P0 finding id."""

    key: str
    columns: Sequence[str]
    reason: str = ""


def resolve_variable_lists(
    categorical_vars: Iterable[str],
    continuous_vars: Iterable[str],
    exclude_cols: Optional[Iterable[str]] = None,
) -> tuple[List[str], List[str]]:
    """Drop ``exclude_cols`` from the two role lists, preserving order.

    Raises
    ------
    ValueError
        If an excluded column does not appear in either role list.  Silently
        ignoring a typo would produce a table that looks like a valid
        sensitivity analysis but is not, so this fails loudly.
    """
    cat = list(categorical_vars)
    cont = list(continuous_vars)
    if not exclude_cols:
        return cat, cont

    excl = list(dict.fromkeys(exclude_cols))
    known = set(cat) | set(cont)
    unknown = [c for c in excl if c not in known]
    if unknown:
        raise ValueError(
            "exclude_cols contains columns absent from the declared roles: "
            f"{unknown}. Known columns: {sorted(known)}"
        )
    excl_set = set(excl)
    return [c for c in cat if c not in excl_set], [c for c in cont if c not in excl_set]


# --------------------------------------------------------------------------- #
# Evaluation entry points
# --------------------------------------------------------------------------- #


def evaluate(
    X_imputed: pd.DataFrame,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    categorical_vars: Sequence[str],
    continuous_vars: Sequence[str],
    *,
    exclude_cols: Optional[Sequence[str]] = None,
    mask_df: Optional[pd.DataFrame] = None,
) -> EvaluationResult:
    """R0-identical evaluation, optionally excluding declared columns.

    With ``exclude_cols=None`` this is a pure pass-through to
    :func:`sni.metrics.evaluate_imputation` and therefore reproduces the R0
    numbers bit for bit.
    """
    cat, cont = resolve_variable_lists(categorical_vars, continuous_vars, exclude_cols)
    return evaluate_imputation(
        X_imputed=X_imputed,
        X_complete=X_complete,
        X_missing=X_missing,
        categorical_vars=cat,
        continuous_vars=cont,
        mask_df=mask_df,
    )


def evaluate_many(
    X_imputed: pd.DataFrame,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    specs: Sequence[EvaluationSpec],
    *,
    mask_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score one imputation under several column-exclusion specs.

    Returns a tidy frame with one row per (spec, metric).  This is the shape
    T1.6a consumes: "metric under the published column set" vs "metric with the
    problem columns dropped", computed offline from a stored ``imputed.csv``
    with zero model re-runs.
    """
    rows: List[Dict[str, object]] = []
    for spec in specs:
        res = evaluate(
            X_imputed,
            X_complete,
            X_missing,
            spec.categorical_vars,
            spec.continuous_vars,
            exclude_cols=spec.exclude_cols,
            mask_df=mask_df,
        )
        for metric, value in res.summary.items():
            rows.append(
                {
                    "spec": spec.name,
                    "excluded": "|".join(spec.exclude_cols),
                    "metric": metric,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def evaluate_run_directory(
    run_dir: str | Path,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    specs: Sequence[EvaluationSpec],
    *,
    imputed_filename: str = "imputed.csv",
    mask_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Offline re-scoring of a stored R0 run.

    R0 kept ``imputed.csv`` for every one of the 1,430 successful runs, so any
    metric-definition change (including column exclusion) can be recomputed
    without touching a model.
    """
    run_dir = Path(run_dir)
    imputed_path = run_dir / imputed_filename
    if not imputed_path.exists():
        raise FileNotFoundError(f"no imputed matrix at {imputed_path}")
    X_imputed = pd.read_csv(imputed_path)
    if len(X_imputed) != len(X_complete):
        raise ValueError(
            f"row-count mismatch: imputed={len(X_imputed)} complete={len(X_complete)}"
        )
    X_imputed.index = X_complete.index
    out = evaluate_many(X_imputed, X_complete, X_missing, specs, mask_df=mask_df)
    out.insert(0, "run", run_dir.name)
    return out


def summary_delta(
    baseline_summary: Mapping[str, float],
    variant_summary: Mapping[str, float],
) -> pd.DataFrame:
    """Tabulate ``variant - baseline`` for every shared summary key."""
    keys = [k for k in baseline_summary if k in variant_summary]
    rows = []
    for k in keys:
        b = float(baseline_summary[k])
        v = float(variant_summary[k])
        rows.append(
            {
                "metric": k,
                "baseline": b,
                "variant": v,
                "delta": v - b,
                "rel_delta": (v - b) / b if b not in (0.0,) and np.isfinite(b) else np.nan,
            }
        )
    return pd.DataFrame(rows)
