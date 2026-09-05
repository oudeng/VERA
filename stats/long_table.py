"""Tidy long-format results table (T1.7, part 2).

Builds ``dataset x mechanism x rate x method x seed x metric x value`` from the
R0 per-seed aggregates ``project_sni_R0/results_all/agg_*/summary_all.csv``.

Why a builder is needed at all
------------------------------
R0 never produced a long table.  What exists is eight per-seed **wide** tables
with mutually incompatible schemas, plus one already-aggregated merge
(``results_all/merged/merged_summary_agg.csv``) that has no seed dimension and
is contaminated (see below).  Six inconsistencies have to be reconciled:

=====  =========================================================================
issue  reconciliation
=====  =========================================================================
1      SNI-side tables carry ``variant``, baseline-side tables carry ``method``.
       Both already derive a unified ``algo``; we keep ``algo`` and also expose
       the raw label as ``method_raw``.
2      ``rate`` is the string ``'30per'`` on the main grid but a float in the
       ext tables.  ``rate_float`` already exists everywhere, so ``rate_float``
       becomes canonical and ``rate`` is re-derived as a normalized label.
3      ``agg_baselines_deep`` **must** be excluded -- all 300 runs failed
       (P0 finding B21).  Enforced by assertion, not convention.
4      ``agg_baselines_deep``'s schema is entirely different (14 cols, no
       metrics), so it cannot be concatenated even if one wanted to.
5      the nine metric columns are melted to (metric, value) pairs.
6      the lambda-ablation rows carry their level in ``exp_body``
       (``lam0.1`` ... ``lam5.0``) while ``variant``/``algo`` both say ``SNI``;
       they get ``experiment_family == 'lambda_ablation'`` and a
       ``method`` of ``lam0.1``, so they can never be silently pooled with the
       learned-lambda main-grid SNI runs.
=====  =========================================================================

Why ``agg_baselines_deep`` is fatal
-----------------------------------
All 300 of its runs died with ``CUDA error: CUDA-capable device(s) is/are busy
or unavailable`` (the single GPU was held by SNI).  It contains no metric
columns at all, and its ``runtime_sec`` values are the ~0.007-30 s it took CUDA
initialization to crash -- e.g. AutoMPG/MAR/30per/GAIN reports **0.0073 s**
against 33.06 s in ``agg_baselines_main``.  Its ``exp_id`` values collide with
``agg_baselines_main`` / ``agg_baselines_mnar``, so concatenating it would give
every GAIN/MIWAE cell ten rows of which five are empty shells, and would poison
any runtime analysis.  ``results_all/merged/merged_summary_agg.csv`` did not
exclude it and consequently logs 65 merge conflicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AGG_SOURCES",
    "EXCLUDED_SOURCES",
    "METRIC_COLUMNS",
    "METRIC_DIRECTION",
    "CANONICAL_METRICS",
    "SourceSpec",
    "assert_deep_excluded",
    "build_long_table",
    "audit_long_table",
    "coverage_table",
    "main_grid_view",
    "to_setting_matrix",
]


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpec:
    """One ``agg_*`` directory and the experiment family it represents."""

    directory: str
    experiment_family: str
    arm: str  # 'sni' | 'baseline'
    note: str = ""


AGG_SOURCES: Mapping[str, SourceSpec] = {
    s.directory: s
    for s in (
        SourceSpec("agg_sni_v03_main", "main_grid", "sni", "SNI, MCAR/MAR @30%"),
        SourceSpec("agg_baselines_main", "main_grid", "baseline", "6 classic baselines, MCAR/MAR @30%"),
        SourceSpec("agg_baselines_new", "main_grid", "baseline", "HyperImpute + TabCSDI, MCAR/MAR @30%"),
        SourceSpec("agg_sni_rate_sweep", "rate_sweep", "sni", "SNI only, MCAR/MAR @10%/50%"),
        SourceSpec("agg_sni_mnar", "mnar", "sni", "SNI + SNI-M, MNAR @10/30/50%"),
        SourceSpec("agg_baselines_mnar", "mnar", "baseline", "6 classic baselines, MNAR @10/30/50%"),
        SourceSpec("agg_sni_v03_ablation_lambda", "lambda_ablation", "sni", "fixed-lambda ablation"),
    )
}

#: Directories that must never enter the long table, with the reason.
EXCLUDED_SOURCES: Mapping[str, str] = {
    "agg_baselines_deep": (
        "P0 finding B21: all 300 runs failed with "
        "'CUDA error: CUDA-capable device(s) is/are busy or unavailable'. "
        "The table has no metric columns, its runtime_sec values are the "
        "crash-time of CUDA initialization (e.g. 0.0073 s vs 33.06 s in "
        "agg_baselines_main), and its exp_id values collide with "
        "agg_baselines_main / agg_baselines_mnar."
    )
}

#: wide column -> canonical metric name
METRIC_COLUMNS: Mapping[str, str] = {
    "cont_NRMSE": "NRMSE",
    "cont_RMSE": "RMSE",
    "cont_MAE": "MAE",
    "cont_MB": "MB",
    "cont_R2": "R2",
    "cont_Spearman": "Spearman",
    "cat_Accuracy": "Accuracy",
    "cat_Macro-F1": "Macro-F1",
    "cat_Cohen_kappa": "Cohen_kappa",
}

CANONICAL_METRICS: Sequence[str] = tuple(METRIC_COLUMNS.values())

#: True = higher is better.  Mirrors ``exp5_significance_tests.py:86-93`` and
#: extends it to the remaining metrics.  ``MB`` (mean bias) is signed and has no
#: monotone direction, hence ``None``.
METRIC_DIRECTION: Mapping[str, Optional[bool]] = {
    "NRMSE": False,
    "RMSE": False,
    "MAE": False,
    "MB": None,
    "R2": True,
    "Spearman": True,
    "Accuracy": True,
    "Macro-F1": True,
    "Cohen_kappa": True,
    "runtime_sec": False,
}

_RATE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*per\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def assert_deep_excluded(sources: Iterable[str]) -> None:
    """Hard guard: refuse to build a long table that includes a failed source.

    Raises
    ------
    ValueError
        If any excluded directory is present in ``sources``.
    """
    bad = [s for s in sources if Path(str(s)).name in EXCLUDED_SOURCES]
    if bad:
        reasons = "; ".join(f"{Path(str(b)).name}: {EXCLUDED_SOURCES[Path(str(b)).name]}" for b in bad)
        raise ValueError(f"excluded result source requested -- {reasons}")


def _normalise_rate(row_rate, row_rate_float) -> tuple[float, str]:
    """Return ``(rate_float, rate_label)`` from either representation."""
    rf = np.nan
    if row_rate_float is not None and not (isinstance(row_rate_float, float) and np.isnan(row_rate_float)):
        try:
            rf = float(row_rate_float)
        except (TypeError, ValueError):
            rf = np.nan
    if not np.isfinite(rf):
        s = str(row_rate)
        m = _RATE_RE.match(s)
        if m:
            rf = float(m.group(1)) / 100.0
        else:
            try:
                rf = float(s)
            except (TypeError, ValueError):
                rf = np.nan
    if not np.isfinite(rf):
        return np.nan, "unknown"
    pct = rf * 100.0
    label = f"{int(round(pct))}per" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}per"
    return rf, label


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_long_table(
    results_root: str | Path,
    *,
    sources: Optional[Sequence[str]] = None,
    include_runtime: bool = True,
    drop_na_values: bool = True,
) -> pd.DataFrame:
    """Build the tidy long table from R0's per-seed aggregates.

    Parameters
    ----------
    results_root
        ``project_sni_R0/results_all``.
    sources
        Directory names to read.  Defaults to every entry of
        :data:`AGG_SOURCES`.  Passing an excluded directory raises.
    include_runtime
        Emit ``metric == 'runtime_sec'`` rows alongside the nine quality
        metrics (needed for the R2-6c cost discussion).
    drop_na_values
        Drop rows whose value is NaN (e.g. a dataset with no categorical
        columns produces no ``cat_*`` metrics).

    Returns
    -------
    pandas.DataFrame
        Columns: ``dataset, mechanism, rate, rate_float, method, algo,
        method_raw, seed, metric, value, experiment_family, source, exp_id,
        higher_is_better``.
    """
    root = Path(results_root)
    if sources is None:
        sources = list(AGG_SOURCES.keys())
    sources = [str(s) for s in sources]

    # E3-style hard guard, executed before any I/O.
    assert_deep_excluded(sources)

    frames: List[pd.DataFrame] = []
    for src in sources:
        spec = AGG_SOURCES.get(src)
        if spec is None:
            raise KeyError(f"unknown result source {src!r}; known: {sorted(AGG_SOURCES)}")
        path = root / src / "summary_all.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing per-seed aggregate: {path}")
        frames.append(_load_one(path, spec, include_runtime=include_runtime))

    long = pd.concat(frames, ignore_index=True)

    if drop_na_values:
        long = long[np.isfinite(long["value"].to_numpy(dtype=float))].copy()

    long["higher_is_better"] = long["metric"].map(METRIC_DIRECTION)
    long = long.sort_values(
        ["experiment_family", "dataset", "mechanism", "rate_float", "method", "seed", "metric"]
    ).reset_index(drop=True)

    # A last-line defense: the fake runtimes of agg_baselines_deep are all
    # sub-second; any real GAIN/MIWAE run takes tens of seconds.
    if include_runtime:
        rt = long[(long["metric"] == "runtime_sec") & (long["algo"].isin(["GAIN", "MIWAE"]))]
        if len(rt) and float(rt["value"].min()) < 1.0:
            raise AssertionError(
                "sub-second GAIN/MIWAE runtime detected -- a failed "
                "agg_baselines_deep row has leaked into the long table"
            )
    return long


def _load_one(path: Path, spec: SourceSpec, *, include_runtime: bool) -> pd.DataFrame:
    df = pd.read_csv(path)

    if spec.arm == "sni":
        if "variant" not in df.columns:
            raise ValueError(f"{path}: SNI-side table lacks the 'variant' column")
        method_raw = df["variant"].astype(str)
    else:
        if "method" not in df.columns:
            raise ValueError(f"{path}: baseline-side table lacks the 'method' column")
        method_raw = df["method"].astype(str)

    if "algo" not in df.columns:
        raise ValueError(f"{path}: no unified 'algo' column")

    # Reconciliation 6: the lambda ablation encodes its level in exp_body.
    if spec.experiment_family == "lambda_ablation":
        if "exp_body" not in df.columns:
            raise ValueError(f"{path}: lambda ablation table lacks 'exp_body'")
        method = df["exp_body"].astype(str)
        bad = [m for m in method.unique() if not str(m).startswith("lam")]
        if bad:
            raise ValueError(f"{path}: unexpected lambda-ablation levels {bad}")
    else:
        method = df["algo"].astype(str)

    rate_float, rate_label = zip(
        *[
            _normalise_rate(r, rf)
            for r, rf in zip(
                df.get("rate", pd.Series([np.nan] * len(df))),
                df.get("rate_float", pd.Series([np.nan] * len(df))),
            )
        ]
    )

    seed = df["seed_parsed"] if "seed_parsed" in df.columns else df["seed"]

    base = pd.DataFrame(
        {
            "dataset": df["dataset"].astype(str),
            "mechanism": df["mechanism"].astype(str),
            "rate": list(rate_label),
            "rate_float": list(rate_float),
            "method": method.values,
            "algo": df["algo"].astype(str).values,
            "method_raw": method_raw.values,
            "seed": pd.to_numeric(seed, errors="coerce").astype("Int64").values,
            "experiment_family": spec.experiment_family,
            "source": spec.directory,
            "exp_id": df["exp_id"].astype(str).values if "exp_id" in df.columns else "",
        }
    )

    value_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    missing_metrics = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing_metrics:
        raise ValueError(
            f"{path}: missing metric columns {missing_metrics} -- this table is not a "
            "successful per-seed aggregate"
        )

    melted = base.join(df[value_cols]).melt(
        id_vars=list(base.columns), value_vars=value_cols, var_name="metric_col", value_name="value"
    )
    melted["metric"] = melted["metric_col"].map(METRIC_COLUMNS)
    melted = melted.drop(columns=["metric_col"])

    if include_runtime and "runtime_sec" in df.columns:
        rt = base.copy()
        rt["metric"] = "runtime_sec"
        rt["value"] = pd.to_numeric(df["runtime_sec"], errors="coerce").values
        melted = pd.concat([melted, rt], ignore_index=True)

    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    return melted


# --------------------------------------------------------------------------- #
# Coverage / audit helpers
# --------------------------------------------------------------------------- #


def audit_long_table(long: pd.DataFrame) -> Dict[str, object]:
    """Structural audit: shape, coverage, duplicates, seeds per cell."""
    cell_keys = ["experiment_family", "dataset", "mechanism", "rate", "method"]
    per_cell = (
        long[long["metric"] == "NRMSE"]
        .groupby(cell_keys, dropna=False)["seed"]
        .agg(["nunique", "size"])
        .reset_index()
    )
    dup = long.duplicated(
        subset=["experiment_family", "dataset", "mechanism", "rate", "method", "seed", "metric"]
    ).sum()
    return {
        "n_rows": int(len(long)),
        "n_cells": int(len(per_cell)),
        "n_runs": int(long.groupby(["source", "exp_id"]).ngroups),
        "n_datasets": int(long["dataset"].nunique()),
        "n_methods": int(long["method"].nunique()),
        "n_metrics": int(long["metric"].nunique()),
        "seeds": sorted(int(s) for s in long["seed"].dropna().unique()),
        "seeds_per_cell": sorted(int(v) for v in per_cell["nunique"].unique()),
        "cells_with_wrong_seed_count": int((per_cell["nunique"] != 5).sum()),
        "duplicate_key_rows": int(dup),
        "sources": sorted(long["source"].unique()),
        "excluded_sources": dict(EXCLUDED_SOURCES),
    }


def coverage_table(long: pd.DataFrame) -> pd.DataFrame:
    """Method x (mechanism, rate) coverage in number of datasets."""
    sub = long[long["metric"] == "NRMSE"]
    piv = (
        sub.groupby(["method", "mechanism", "rate"])["dataset"]
        .nunique()
        .unstack(["mechanism", "rate"])
        .fillna(0)
        .astype(int)
    )
    return piv.sort_index()


def main_grid_view(
    long: pd.DataFrame,
    *,
    mechanisms: Sequence[str] = ("MCAR", "MAR"),
    rate: float = 0.30,
) -> pd.DataFrame:
    """The published main grid: 6 datasets x {MCAR, MAR} x 30% x 9 methods x 5 seeds."""
    return long[
        (long["experiment_family"] == "main_grid")
        & (long["mechanism"].isin(list(mechanisms)))
        & (np.isclose(long["rate_float"].astype(float), float(rate)))
    ].copy()


def to_setting_matrix(
    long: pd.DataFrame,
    metric: str,
    *,
    setting_keys: Sequence[str] = ("dataset", "mechanism"),
    methods: Optional[Sequence[str]] = None,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Collapse seeds and pivot to a ``setting x method`` matrix.

    This is the block structure the Friedman test consumes: rows are the paired
    units (settings), columns the treatments (methods).  Rows with any missing
    method are dropped, because Friedman requires a complete block design.
    """
    sub = long[long["metric"] == metric]
    if methods is not None:
        sub = sub[sub["method"].isin(list(methods))]
    mat = sub.pivot_table(
        index=list(setting_keys), columns="method", values="value", aggfunc=aggfunc
    )
    if methods is not None:
        keep = [m for m in methods if m in mat.columns]
        mat = mat[keep]
    return mat.dropna(axis=0, how="any")
