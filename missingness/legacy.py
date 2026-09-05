"""Bit-exact reproduction of the R0 mask generator.

This is a **behavior-preserving port** of
``project_sni_R0/sni/utility_missing_data_gen_v1/missing_data_generator.py``.
Not a fix, not a cleanup: the defects are the point. It exists so that

* the R0 masks shipped in ``project_sni_R0/sni/data/*/*_mask.npy`` can be
  regenerated inside ``code_SNI`` without depending on the frozen read-only R0
  tree, and
* "old mask vs new mask" comparisons in the P2 rebuild are exact rather than
  approximate — any metric difference is attributable to the mechanism change
  and not to a reimplementation artifact.

Verified bit-identical to the 18 stored R0 masks (6 datasets x 3 mechanisms at
30%) by ``tests/test_missingness.py::test_legacy_port_reproduces_R0_masks_bitwise``.

Reproducing R0 exactly requires reproducing R0's *RNG consumption order*: one
``np.random.default_rng(seed)`` threaded through the MNAR categorical draw, the
whole-table Bernoulli sample, the min-per-column tie-break and the calibration.
That single shared stream is finding B45, and it is why this module cannot share
code with :mod:`missingness.generator`, which deliberately uses independent
per-column streams.

Line references below point at the original file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .spec import DatasetSchema, dataset_schema


@dataclass
class LegacyColumnTypes:
    """R0 ``missing_data_generator.ColumnTypes`` (:41-49)."""
    continuous: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {"continuous": list(self.continuous),
                "categorical": list(self.categorical),
                "excluded": list(self.excluded)}


def column_types_from_schema(schema: DatasetSchema, extra_excluded: Sequence[str] = ()) -> LegacyColumnTypes:
    """Derive R0-shaped column types from ``configs/datasets.yaml``.

    CAVEAT recorded during T1.5: for ``Concrete`` this does NOT match what R0
    actually used. ``configs/datasets.yaml`` declares ``Duration`` continuous
    (14 integer levels, range 1-365) whereas R0's dtype heuristic
    (``missing_data_generator.infer_column_types``, :179-185: integer dtype with
    ``nunique <= 20`` becomes categorical) classified it categorical, and the
    stored ``Concrete_*_meta.json`` confirm ``"categorical": ["Duration"]``.
    Pass ``column_types`` explicitly (e.g. read back from the R0 meta.json) when
    bit-identity matters. Flagged for the first author; not resolved here.
    """
    ex = list(dict.fromkeys([c.name for c in schema.columns.values() if c.is_identifier] + list(extra_excluded)))
    ex_set = set(ex)
    return LegacyColumnTypes(
        continuous=[c for c in schema.continuous() if c not in ex_set],
        categorical=[c for c in schema.categorical() if c not in ex_set],
        excluded=ex,
    )


# ---------------------------------------------------------------------------
# verbatim ports
# ---------------------------------------------------------------------------

def _cast_dataframe_for_generation(df: pd.DataFrame, ct: LegacyColumnTypes) -> pd.DataFrame:
    """``missing_data_generator.cast_dataframe_for_generation`` (:215-268)."""
    out = df.copy()
    for c in ct.continuous:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ct.categorical:
        if c not in out.columns:
            continue
        s = out[c]
        num = pd.to_numeric(s, errors="coerce")
        non_na = num.dropna()
        if len(non_na) > 0 and bool(((non_na - non_na.round()).abs() <= 1e-8).all()):
            out[c] = num.round().astype("Int64")
        else:
            out[c] = s.astype("category")
    for c in ct.excluded:
        if c not in out.columns:
            continue
        num = pd.to_numeric(out[c], errors="coerce")
        non_na = num.dropna()
        if len(non_na) == 0:
            continue
        if bool(((non_na - non_na.round()).abs() <= 1e-8).all()):
            out[c] = num.round().astype("Int64")
    return out


def _standardize_series(x: pd.Series) -> np.ndarray:
    """``missing_data_generator._standardize_series`` (:425-430)."""
    arr = x.to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    sd = sd if sd > 1e-12 else 1.0
    return (arr - mu) / sd


def _mcar_propensity(df: pd.DataFrame, rate: float, exclude_mask: np.ndarray) -> np.ndarray:
    """``missing_data_generator._mcar_propensity`` (:418-422)."""
    p = np.full(df.shape, float(rate), dtype=float)
    p[exclude_mask] = 0.0
    return p


def _mar_propensity(df: pd.DataFrame, rate: float, driver_cols: Sequence[str],
                    exclude_mask: np.ndarray, logistic_scale: float = 1.0) -> np.ndarray:
    """``missing_data_generator._mar_propensity`` (:433-470).

    The two defects, unchanged: ``p_row * (rate / mean_p)`` instead of an
    intercept fit (:460-464), and ``np.repeat`` broadcasting one row propensity
    onto every column (:467).
    """
    if len(driver_cols) == 0:
        raise ValueError("MAR requires at least one driver column.")
    z_sum = None
    for c in driver_cols:
        z = _standardize_series(df[c])
        z_sum = z if z_sum is None else (z_sum + z)
    lin = logistic_scale * z_sum
    p_row = 1.0 / (1.0 + np.exp(-lin))
    mean_p = float(np.mean(p_row))
    if mean_p <= 1e-12:
        p_row = np.full_like(p_row, rate, dtype=float)
    else:
        p_row = p_row * (rate / mean_p)
    p_row = np.clip(p_row, 0.0, 1.0)
    p = np.repeat(p_row[:, None], df.shape[1], axis=1)
    p[exclude_mask] = 0.0
    return p


def _mnar_propensity(df: pd.DataFrame, rate: float, ct: LegacyColumnTypes,
                     rng: np.random.Generator, exclude_mask: np.ndarray,
                     q_low: float = 0.25, q_mid: float = 0.50, q_high: float = 0.75,
                     p_low_mult: float = 0.4, p_mid_low_mult: float = 0.8,
                     p_mid_high_mult: float = 1.2, p_high_mult: float = 1.8,
                     cat_high_frac: float = 0.5, cat_high_mult: float = 1.5,
                     cat_low_mult: float = 0.7) -> np.ndarray:
    """``missing_data_generator._mnar_propensity`` (:473-559).

    Continuous columns get the 4-quantile step function (:505-526); categorical
    columns get a random ``cat_high_frac`` of their levels marked high-missing
    (:529-553), drawn from the *shared* generator — B45 — and with no intercept
    fit, so a single-level column lands every row at ``rate * cat_high_mult``
    (B46) and the column overshoots its target (B39).
    """
    n_rows, n_cols = df.shape
    p = np.zeros((n_rows, n_cols), dtype=float)

    for col in ct.continuous:
        j = df.columns.get_loc(col)
        values = df[col].to_numpy(dtype=float)
        q1 = np.nanpercentile(values, q_low * 100.0)
        q2 = np.nanpercentile(values, q_mid * 100.0)
        q3 = np.nanpercentile(values, q_high * 100.0)
        probs = np.empty_like(values, dtype=float)
        probs[values > q3] = rate * p_high_mult
        probs[(values > q2) & (values <= q3)] = rate * p_mid_high_mult
        probs[(values > q1) & (values <= q2)] = rate * p_mid_low_mult
        probs[values <= q1] = rate * p_low_mult
        p[:, j] = np.clip(probs, 0.0, 1.0)

    for col in ct.categorical:
        j = df.columns.get_loc(col)
        s = df[col]
        uniques = pd.unique(s.astype("object"))
        uniques = [u for u in uniques if pd.notna(u)]
        if len(uniques) == 0:
            p[:, j] = rate
            continue
        n_high = max(1, int(round(len(uniques) * cat_high_frac)))
        high_cats = set(rng.choice(uniques, size=n_high, replace=False).tolist())
        probs = np.empty(n_rows, dtype=float)
        vals = s.astype("object").to_numpy()
        for i, v in enumerate(vals):
            if pd.isna(v):
                probs[i] = rate
            elif v in high_cats:
                probs[i] = rate * cat_high_mult
            else:
                probs[i] = rate * cat_low_mult
        p[:, j] = np.clip(probs, 0.0, 1.0)

    p[exclude_mask] = 0.0
    return p


def _enforce_min_missing_per_column(mask: np.ndarray, propensity: np.ndarray,
                                    min_missing_per_col: int, rng: np.random.Generator,
                                    excluded_cols_mask: np.ndarray) -> np.ndarray:
    """``missing_data_generator._enforce_min_missing_per_column`` (:271-321)."""
    n_rows, n_cols = mask.shape
    if min_missing_per_col <= 0:
        return mask
    new_mask = mask.copy()
    for j in range(n_cols):
        if bool(excluded_cols_mask[j]):
            continue
        cur = int(new_mask[:, j].sum())
        if cur >= min_missing_per_col:
            continue
        need = min_missing_per_col - cur
        cand = np.where(~new_mask[:, j])[0]
        if cand.size == 0:
            continue
        pj = propensity[cand, j]
        noise = rng.random(cand.size) * 1e-12
        order = np.argsort(-(pj + noise))
        new_mask[cand[order[: min(need, cand.size)]], j] = True
    return new_mask


def _calibrate_mask_to_rate(mask: np.ndarray, propensity: np.ndarray, target_rate: float,
                            tolerance: float, rng: np.random.Generator,
                            exclude_mask: np.ndarray) -> np.ndarray:
    """``missing_data_generator.calibrate_mask_to_rate`` (:324-411).

    Table-level only. This is what let per-column rates drift to 37.8% / 43.4%
    while the eligible-cell total sat on 30.0% — finding B39.
    """
    eligible = ~exclude_mask
    total = int(eligible.sum())
    if total == 0:
        return mask
    target = int(round(float(target_rate) * total))
    new_mask = mask.copy()
    new_mask[exclude_mask] = False
    cur = int(new_mask[eligible].sum())
    if abs(cur / total - float(target_rate)) <= float(tolerance):
        return new_mask

    if cur < target:
        k = target - cur
        cand_pos = np.where((~new_mask & eligible).ravel())[0]
        if cand_pos.size == 0:
            return new_mask
        p = np.clip(propensity.ravel()[cand_pos].astype(float), 0.0, 1.0)
        if float(p.sum()) <= 0.0:
            chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False)
        else:
            chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False, p=p / p.sum())
        new_mask.ravel()[chosen] = True
        new_mask[exclude_mask] = False
        return new_mask

    k = cur - target
    cand_pos = np.where((new_mask & eligible).ravel())[0]
    if cand_pos.size == 0:
        return new_mask
    p = np.clip(propensity.ravel()[cand_pos].astype(float), 0.0, 1.0)
    score = 1.0 - p
    if float(score.sum()) <= 0.0:
        chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False)
    else:
        chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False, p=score / score.sum())
    new_mask.ravel()[chosen] = False
    new_mask[exclude_mask] = False
    return new_mask


# ---------------------------------------------------------------------------

@dataclass
class LegacyResult:
    X_missing: pd.DataFrame
    mask: np.ndarray            # bool, True = missing
    column_types: LegacyColumnTypes
    meta: Dict[str, Any]


def generate_legacy(
    df: pd.DataFrame,
    *,
    dataset: str,
    mechanism: str,
    rate: float,
    seed: int = 2025,
    column_types: Optional[LegacyColumnTypes] = None,
    mar_driver_cols: Sequence[str] = ("ID",),
    mar_logistic_scale: float = 1.0,
    tolerance: float = 0.01,
    min_missing_per_col: int = 1,
    schema: Optional[DatasetSchema] = None,
) -> LegacyResult:
    """Reproduce one R0 mask.

    ``mar_driver_cols`` defaults to ``("ID",)`` because that is literally what R0
    used for all six datasets — every ``*_MAR_*_meta.json`` records
    ``"mar_driver_cols": ["ID"]``. ``ID`` is a 1..n record counter, which is the
    substance of reviewer point R1-4.
    """
    mechanism = mechanism.strip().upper()
    if mechanism not in {"MCAR", "MAR", "MNAR"}:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    ct = column_types or column_types_from_schema(schema or dataset_schema(dataset))
    if mechanism == "MAR":
        ct = LegacyColumnTypes(
            continuous=[c for c in ct.continuous if c not in set(mar_driver_cols)],
            categorical=[c for c in ct.categorical if c not in set(mar_driver_cols)],
            excluded=list(dict.fromkeys(list(ct.excluded) + list(mar_driver_cols))),
        )

    rng = np.random.default_rng(seed)   # THE shared stream (B45)
    df_cast = _cast_dataframe_for_generation(df, ct)

    exclude_mask = np.zeros(df_cast.shape, dtype=bool)
    excluded_cols_mask = np.zeros(df_cast.shape[1], dtype=bool)
    for col in ct.excluded:
        j = df_cast.columns.get_loc(col)
        exclude_mask[:, j] = True
        excluded_cols_mask[j] = True

    if mechanism == "MCAR":
        propensity = _mcar_propensity(df_cast, rate, exclude_mask)
    elif mechanism == "MAR":
        propensity = _mar_propensity(df_cast, rate, list(mar_driver_cols), exclude_mask,
                                     logistic_scale=mar_logistic_scale)
    else:
        propensity = _mnar_propensity(df_cast, rate, ct, rng, exclude_mask)

    mask = rng.random(df_cast.shape) < propensity
    mask[exclude_mask] = False
    mask = _enforce_min_missing_per_column(mask, propensity, min_missing_per_col, rng, excluded_cols_mask)
    mask[exclude_mask] = False
    mask = _calibrate_mask_to_rate(mask, propensity, rate, tolerance, rng, exclude_mask)
    mask[exclude_mask] = False
    mask = _enforce_min_missing_per_column(mask, propensity, min_missing_per_col, rng, excluded_cols_mask)
    mask[exclude_mask] = False

    X_missing = df_cast.mask(mask)
    eligible = ~exclude_mask
    meta = {
        "generator": "code_SNI/missingness/legacy.py (verbatim R0 port)",
        "dataset_name": dataset,
        "mechanism": mechanism,
        "target_rate": float(rate),
        "actual_rate_eligible": float(mask[eligible].mean()) if eligible.sum() else 0.0,
        "actual_rate_all": float(X_missing.isna().to_numpy().mean()),
        "seed": int(seed),
        "column_types": ct.as_dict(),
        "mar_driver_cols": list(mar_driver_cols) if mechanism == "MAR" else [],
        "mar_logistic_scale": float(mar_logistic_scale),
        "tolerance": float(tolerance),
        "min_missing_per_col": int(min_missing_per_col),
        "per_column_missing_rate": {c: float(X_missing[c].isna().mean()) for c in X_missing.columns},
        "WARNING": ("R0 reproduction with all known defects intact "
                    "(row-broadcast MAR, mean-linear rescale, shared RNG, table-only "
                    "calibration). Not for new results."),
    }
    return LegacyResult(X_missing=X_missing, mask=mask, column_types=ct, meta=meta)
