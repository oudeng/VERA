from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# SciPy emits ConstantInputWarning when either input to spearmanr is constant.
# We want metrics to be numerically stable (rho=0.0) and log-clean.
try:  # pragma: no cover
    from scipy.stats import ConstantInputWarning  # type: ignore
except Exception:  # pragma: no cover
    ConstantInputWarning = Warning  # type: ignore
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def nrmse(true: np.ndarray, pred: np.ndarray, value_range: Optional[float] = None) -> float:
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    if value_range is None:
        value_range = float(np.nanmax(true) - np.nanmin(true))
    return rmse / value_range if value_range and value_range > 0 else rmse


def spearman_rho(true: np.ndarray, pred: np.ndarray) -> float:
    """Spearman's rho on 1-D arrays.

    Spearman is undefined when either input is constant. We treat such cases as
    rho=0.0 (conservative) and avoid emitting scipy warnings.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if true.size < 2:
        return float("nan")

    # Avoid SciPy ConstantInputWarning & NaN propagation.
    std_true = float(np.nanstd(true))
    std_pred = float(np.nanstd(pred))
    if (not np.isfinite(std_true)) or (not np.isfinite(std_pred)):
        return 0.0
    if std_true < 1e-12 or std_pred < 1e-12:
        return 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            # nan_policy='omit' prevents NaNs from making rho undefined.
            rho, _ = spearmanr(true, pred, nan_policy="omit")
        except TypeError:
            # Backward compatibility for older SciPy without nan_policy.
            rho, _ = spearmanr(true, pred)

    if np.isnan(rho) or (not np.isfinite(rho)):
        return 0.0
    return float(rho)


def compute_continuous_metrics(true_vals: np.ndarray, pred_vals: np.ndarray, full_true_col: Optional[pd.Series] = None) -> Dict[str, float]:
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(true_vals, pred_vals)))
    mae = float(mean_absolute_error(true_vals, pred_vals))
    r2 = float(r2_score(true_vals, pred_vals)) if len(true_vals) >= 2 else float("nan")

    if full_true_col is not None:
        vr = float(full_true_col.max() - full_true_col.min())
    else:
        vr = float(np.nanmax(true_vals) - np.nanmin(true_vals))
    nrmse_val = nrmse(true_vals, pred_vals, vr)

    mb = float(np.mean(pred_vals - true_vals))
    rho = spearman_rho(true_vals, pred_vals)

    return {"RMSE": rmse, "NRMSE": nrmse_val, "MAE": mae, "MB": mb, "R2": r2, "Spearman": rho}


def _maybe_round_to_int(true_s: pd.Series, pred_s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    '''
    If ground-truth looks integer-coded, round predictions to nearest int.
    '''
    true_num = _safe_numeric_series(true_s)
    pred_num = _safe_numeric_series(pred_s)

    if true_num.notna().mean() > 0.8:
        # Heuristic: most values are numeric; check integer-likeness
        frac = np.abs(true_num.dropna() - np.round(true_num.dropna()))
        if (frac < 1e-6).mean() > 0.95:
            return true_num.round().astype("Int64"), pred_num.round().astype("Int64")

    # fallback as strings
    return true_s.astype(str), pred_s.astype(str)


def compute_categorical_metrics(
    true_s: pd.Series,
    pred_s: pd.Series,
    *,
    all_labels: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute categorical metrics on aligned (true, pred) series.

    Notes
    -----
    - We pass an explicit ``labels`` list to sklearn metrics to avoid warnings
      when only a single class appears in the evaluation subset.
    - Cohen's kappa is undefined for degenerate single-class problems; we
      return 0.0 in that case (conservative) and suppress runtime warnings.
    """
    true_aligned, pred_aligned = _maybe_round_to_int(true_s, pred_s)

    # align indices
    common = true_aligned.index.intersection(pred_aligned.index)
    true_vals = true_aligned.loc[common]
    pred_vals = pred_aligned.loc[common]

    # drop NaNs
    mask = true_vals.notna() & pred_vals.notna()
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]

    if len(true_vals) == 0:
        return {"Accuracy": float("nan"), "Macro-F1": float("nan"), "Cohen_kappa": float("nan")}

    # Determine the label set.
    if all_labels is not None:
        labels = pd.Series(all_labels).dropna().unique().tolist()
    else:
        labels = pd.concat([true_vals, pred_vals], axis=0).dropna().unique().tolist()

    # Ensure deterministic order when possible.
    try:
        labels = sorted(labels)
    except Exception:
        pass

    # Many sklearn categorical metrics emit warnings on degenerate evaluation
    # subsets (e.g., a single class appears in y_true/y_pred). These cases are
    # expected when the evaluation set is ``missing positions only`` and the
    # missing mask is small / imbalanced. We silence the warnings and return
    # conservative numeric fallbacks.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        acc = float(accuracy_score(true_vals, pred_vals))
        f1_macro = float(
            f1_score(
                true_vals,
                pred_vals,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

        # Cohen's kappa
        if len(labels) < 2:
            kappa = 0.0
        else:
            kappa = float(cohen_kappa_score(true_vals, pred_vals, labels=labels))
            if np.isnan(kappa) or (not np.isfinite(kappa)):
                kappa = 0.0

    return {"Accuracy": acc, "Macro-F1": f1_macro, "Cohen_kappa": kappa}


@dataclass
class EvaluationResult:
    per_feature: pd.DataFrame
    summary: Dict[str, float]


def augment_summary_with_imputer_stats(summary: Dict, imputer) -> Dict:
    """
    v0.3: Augment metrics summary dict with imputer runtime/convergence/lambda stats.

    Args:
        summary: Existing metrics summary dict
        imputer: SNIImputer instance (after impute() has been called)

    Returns:
        Augmented summary dict
    """
    summary["runtime_sec"] = getattr(imputer, "runtime_sec_", 0.0)
    summary["convergence_iterations"] = getattr(imputer, "convergence_iterations_", 0)

    if not getattr(imputer, "did_converge_", True):
        summary["warning"] = "did_not_converge"

    lam_summary = imputer.get_lambda_summary() if hasattr(imputer, "get_lambda_summary") else {}
    summary["lambda_mean"] = lam_summary.get("lambda_mean", 0.0)
    summary["lambda_std"] = lam_summary.get("lambda_std", 0.0)

    return summary


def evaluate_imputation(
    X_imputed: pd.DataFrame,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    mask_df: Optional[pd.DataFrame] = None,
) -> EvaluationResult:
    '''
    Evaluate only on positions that are missing in X_missing.
    If mask_df is provided, it must be boolean with True at evaluation positions.
    '''
    if mask_df is None:
        mask_df = X_missing.isna()

    rows = []
    # Continuous
    for col in continuous_vars:
        if col not in X_imputed.columns:
            continue
        miss_mask = mask_df[col]
        if int(miss_mask.sum()) == 0:
            continue

        true_s = _safe_numeric_series(X_complete.loc[miss_mask, col]).dropna()
        pred_s = _safe_numeric_series(X_imputed.loc[true_s.index, col])
        if len(true_s) == 0:
            continue

        m = compute_continuous_metrics(true_s.values, pred_s.values, full_true_col=_safe_numeric_series(X_complete[col]))
        m.update({"feature": col, "type": "continuous", "n_eval": int(len(true_s))})
        rows.append(m)

    # Categorical
    for col in categorical_vars:
        if col not in X_imputed.columns:
            continue
        miss_mask = mask_df[col]
        if int(miss_mask.sum()) == 0:
            continue

        true_s = X_complete.loc[miss_mask, col].dropna()
        pred_s = X_imputed.loc[true_s.index, col]
        if len(true_s) == 0:
            continue

        m = compute_categorical_metrics(true_s, pred_s, all_labels=X_complete[col])
        m.update({"feature": col, "type": "categorical", "n_eval": int(len(true_s))})
        rows.append(m)

    per_feature = pd.DataFrame(rows)

    summary: Dict[str, float] = {}
    if not per_feature.empty:
        cont_df = per_feature[per_feature["type"] == "continuous"]
        cat_df = per_feature[per_feature["type"] == "categorical"]

        for metric in ["NRMSE", "RMSE", "MAE", "MB", "R2", "Spearman"]:
            if metric in cont_df.columns and len(cont_df) > 0:
                summary[f"cont_{metric}"] = float(np.nanmean(cont_df[metric].values))
        for metric in ["Accuracy", "Macro-F1", "Cohen_kappa"]:
            if metric in cat_df.columns and len(cat_df) > 0:
                summary[f"cat_{metric}"] = float(np.nanmean(cat_df[metric].values))

        summary["n_cont_features"] = int(len(cont_df))
        summary["n_cat_features"] = int(len(cat_df))
        summary["n_features_total"] = int(len(cont_df) + len(cat_df))

    return EvaluationResult(per_feature=per_feature, summary=summary)
