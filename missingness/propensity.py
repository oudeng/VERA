"""Per-column propensity construction for MCAR / MAR / MNAR.

This module is the substantive rewrite requested by task T1.5. It replaces three
mechanisms in ``project_sni_R0/sni/utility_missing_data_gen_v1/missing_data_generator.py``:

===============================  =========================================  ==========================================
R0                               defect                                     R1 replacement
===============================  =========================================  ==========================================
``_mar_propensity`` (:433-470)   one row-level propensity broadcast to       :func:`mar_propensity` — an independent
                                 every column with ``np.repeat`` (:467)      logistic per target column
``_mar_propensity`` (:460-464)   ``p * (rate/mean_p)`` mean-linear rescale   :func:`calibration.solve_intercept` —
                                                                            logit-space bisection on the intercept
``_mnar_propensity`` (:505-526)  4-quantile step function on the value       :func:`mnar_propensity` mode ``logit`` —
                                                                            continuous logistic on the standardized
                                                                            value (``quantile_steps`` retained)
``_mnar_propensity`` (:529-553)  random 50% of levels declared "high         modes ``ordinal`` / ``semantic_groups`` —
                                 missing" using the shared table RNG         declared structure, private RNG stream
===============================  =========================================  ==========================================

Every function returns ``(p, record)`` where ``record`` is a JSON-ready dict that
lands in ``meta.json``: drivers, coefficients, the fitted intercept, the expected
rate implied by the propensity, and any degeneracy flags.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import fit_logistic_propensity, sigmoid, solve_intercept, standardize
from .spec import ColumnMeta, MARColumnSpec, MissingnessSpec, MNARColumnSpec


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------

def level_order(series: pd.Series, meta: Optional[ColumnMeta] = None,
                explicit: Optional[Sequence[Any]] = None) -> List[Any]:
    """Deterministic ordering of a categorical column's levels.

    Priority: explicit spec ordering > ``levels`` declared in
    ``configs/datasets.yaml`` > sorted unique values. Sorting (rather than
    ``pd.unique``, which is *arrival* order) is what makes the encoding
    independent of row order — R0's ``pd.unique`` at
    ``missing_data_generator.py:533`` fed an arrival-ordered list straight into
    ``rng.choice``, so shuffling the rows changed the mask.
    """
    if explicit:
        return list(explicit)
    if meta is not None and meta.levels:
        return list(meta.levels)
    vals = pd.Series(series).dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:  # mixed types
        return sorted(vals, key=repr)


def encode_column(series: pd.Series, meta: Optional[ColumnMeta] = None,
                  explicit_levels: Optional[Sequence[Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Standardized numeric encoding of any column, plus a description.

    * continuous -> z-score of the value
    * categorical -> z-score of the level's *rank* in :func:`level_order`

    Ordinal ranking is the "declared ordinal/semantic structure" the T1.5 brief
    asks for. It is meaningful for the columns that matter clinically
    (``gcs``, ``age_band``, ``composite_risk_score``, ``ALARM`` severity codes),
    and for a nominal column with no declared order the config should use
    ``semantic_groups`` instead — which is why ``random_split`` had to go.
    """
    is_cat = meta is not None and meta.is_categorical
    if is_cat or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        levels = level_order(series, meta, explicit_levels)
        rank = {lv: i for i, lv in enumerate(levels)}
        codes = pd.Series(series).map(rank).astype(float).to_numpy()
        z, mu, sd = standardize(codes)
        return z, {"encoding": "ordinal_rank_zscore", "levels": [_j(l) for l in levels],
                   "n_levels": len(levels), "mean": mu, "std": sd, "degenerate": sd == 0.0}
    z, mu, sd = standardize(pd.Series(series).to_numpy(dtype=float))
    return z, {"encoding": "zscore", "mean": mu, "std": sd, "degenerate": sd == 0.0}


def _j(x: Any) -> Any:
    """Make a level JSON-serialisable without changing its identity in prose."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _auto_coefficient(rng: np.random.Generator, strength: float, min_abs: float) -> float:
    """Slope drawn uniformly on ``[-strength, strength]``, pushed off zero.

    Verbatim port of ``scripts/synth_generate_s5.py:306-308`` (MAR) and
    ``:331-333`` (MNAR): a slope in ``(-min_abs, min_abs)`` would make the column
    indistinguishable from MCAR, so it is snapped to ``+/-min_abs``. The sign is
    kept, which is what gives the *positive or negative* sensitivity the brief
    requires.
    """
    a = float(rng.uniform(-strength, strength))
    if abs(a) < min_abs:
        a = min_abs * (np.sign(a) if a != 0 else 1.0)
    return float(a)


# ---------------------------------------------------------------------------
# MCAR
# ---------------------------------------------------------------------------

def mcar_propensity(n: int, rate: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = np.full(int(n), float(rate), dtype=float)
    return p, {"mechanism": "MCAR", "p": float(rate), "expected_rate": float(rate)}


# ---------------------------------------------------------------------------
# MAR
# ---------------------------------------------------------------------------

def mar_propensity(
    df: pd.DataFrame,
    column: str,
    col_spec: MARColumnSpec,
    spec: MissingnessSpec,
    rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Independent logistic propensity for ONE target column.

    ``eta_j = sum_k beta_jk * z(driver_k)``  and  ``p_j = sigmoid(eta_j + b_j)``
    with ``b_j`` solved so that ``mean(p_j) == rate`` exactly.

    Three properties R0 did not have:

    1. ``beta_jk`` is indexed by *column and driver*. Two columns fed by the same
       driver can respond in opposite directions.
    2. ``b_j`` comes from bisection, not from rescaling, so the slope — the thing
       that defines the mechanism — survives the rate calibration untouched.
    3. ``rng`` is this column's private ``propensity`` stream, so an ``auto``
       coefficient for column *j* is a function of the column's name only (B45).
    """
    drivers = list(col_spec.drivers)
    eta = np.zeros(len(df), dtype=float)
    coef_record: Dict[str, Any] = {}

    for drv in drivers:
        meta = spec.schema.columns.get(drv)
        coef = col_spec.coefficient_for(drv)

        if isinstance(coef, Mapping):
            # Per-level additive logit offsets for a categorical driver.
            levels = level_order(df[drv], meta)
            off = {(_j(k)): float(v) for k, v in coef.items()}
            contrib = pd.Series(df[drv]).map(lambda v: off.get(_j(v), 0.0)).astype(float).to_numpy()
            eta += contrib
            coef_record[drv] = {"type": "level_offsets", "offsets": off,
                                "levels_present": [_j(l) for l in levels]}
            continue

        z, enc = encode_column(df[drv], meta)
        if coef is None:
            if spec.mar_coefficient_default == "config":
                raise ValueError(
                    f"MAR column {column!r}: no coefficient declared for driver {drv!r} and "
                    f"mar.coefficient_default is 'config'. Declare it in configs/missingness.yaml."
                )
            beta = _auto_coefficient(rng, spec.mar_auto_strength, spec.mar_auto_min_abs)
            source = "auto"
        else:
            beta = float(coef)
            source = "config"
        eta += beta * z
        coef_record[drv] = {"type": "scalar", "beta": beta, "source": source, **enc}

    p, b, expected = fit_logistic_propensity(eta, rate)
    return p, {
        "mechanism": "MAR",
        "drivers": drivers,
        "coefficients": coef_record,
        "intercept": b,
        "target_rate": float(rate),
        "expected_rate": expected,
        "eta_std": float(np.std(eta)),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "degenerate": bool(np.std(eta) == 0.0),
    }


def mar_propensity_legacy_row(
    df: pd.DataFrame,
    drivers: Sequence[str],
    spec: MissingnessSpec,
    rate: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Faithful reproduction of ``missing_data_generator._mar_propensity`` (:433-470).

    Kept ONLY so that R0's masks can be regenerated and compared against the new
    ones. Both defects are present by design: the sum of standardized drivers
    goes through one shared sigmoid, and the level is fixed by the mean-linear
    rescale ``p * (rate / mean_p)`` followed by ``clip(0,1)``.
    """
    z_sum = None
    for c in drivers:
        z = np.asarray(df[c], dtype=float)
        mu, sd = np.nanmean(z), np.nanstd(z)
        sd = sd if sd > 1e-12 else 1.0
        z = (z - mu) / sd
        z_sum = z if z_sum is None else z_sum + z
    lin = spec.mar_logistic_scale * z_sum
    p_row = 1.0 / (1.0 + np.exp(-lin))
    mean_p = float(np.mean(p_row))
    if mean_p <= 1e-12:
        p_row = np.full_like(p_row, float(rate))
    else:
        p_row = p_row * (float(rate) / mean_p)
    p_row = np.clip(p_row, 0.0, 1.0)
    return p_row, {
        "mechanism": "MAR",
        "mode": "row_broadcast_legacy",
        "drivers": list(drivers),
        "logistic_scale": float(spec.mar_logistic_scale),
        "rescale_factor": float(rate) / mean_p if mean_p > 1e-12 else None,
        "expected_rate": float(np.mean(p_row)),
        "warning": "R0 reproduction: identical propensity broadcast to every column",
    }


# ---------------------------------------------------------------------------
# MNAR
# ---------------------------------------------------------------------------

_QSTEP_DEFAULTS = {
    "q_low": 0.25, "q_mid": 0.50, "q_high": 0.75,
    "p_low_mult": 0.4, "p_mid_low_mult": 0.8, "p_mid_high_mult": 1.2, "p_high_mult": 1.8,
    "cat_high_frac": 0.5, "cat_high_mult": 1.5, "cat_low_mult": 0.7,
}


def mnar_propensity(
    df: pd.DataFrame,
    column: str,
    col_spec: MNARColumnSpec,
    spec: MissingnessSpec,
    rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """MNAR propensity for ONE column, driven by that column's own value."""
    meta = spec.schema.columns.get(column)
    s = df[column]
    n = len(df)
    mode = col_spec.mode
    qs = {**_QSTEP_DEFAULTS, **(spec.mnar_quantile_steps or {})}

    # ---- legacy continuous: 4-quantile step function -----------------------
    if mode == "quantile_steps":
        v = pd.Series(s).to_numpy(dtype=float)
        q1 = np.nanpercentile(v, qs["q_low"] * 100.0)
        q2 = np.nanpercentile(v, qs["q_mid"] * 100.0)
        q3 = np.nanpercentile(v, qs["q_high"] * 100.0)
        p = np.empty(n, dtype=float)
        p[v > q3] = rate * qs["p_high_mult"]
        p[(v > q2) & (v <= q3)] = rate * qs["p_mid_high_mult"]
        p[(v > q1) & (v <= q2)] = rate * qs["p_mid_low_mult"]
        p[v <= q1] = rate * qs["p_low_mult"]
        p = np.clip(p, 0.0, 1.0)
        return p, {
            "mechanism": "MNAR", "mode": "quantile_steps",
            "quantiles": {"q1": float(q1), "q2": float(q2), "q3": float(q3)},
            "multipliers": {k: qs[k] for k in
                            ("p_low_mult", "p_mid_low_mult", "p_mid_high_mult", "p_high_mult")},
            "target_rate": float(rate), "expected_rate": float(p.mean()),
            "warning": "R0 comparison mode: step function, no intercept calibration",
        }

    # ---- legacy categorical: random 50% split ------------------------------
    if mode == "random_split":
        vals = pd.Series(s).astype("object")
        uniques = [u for u in pd.unique(vals) if pd.notna(u)]
        if len(uniques) == 0:
            return np.full(n, float(rate)), {
                "mechanism": "MNAR", "mode": "random_split", "degenerate": True,
                "target_rate": float(rate), "expected_rate": float(rate)}
        n_high = max(1, int(round(len(uniques) * qs["cat_high_frac"])))
        high = set(np.asarray(rng.choice(np.asarray(uniques, dtype=object), size=n_high,
                                         replace=False)).tolist())
        p = np.where(vals.isin(high).to_numpy(), rate * qs["cat_high_mult"], rate * qs["cat_low_mult"])
        p = np.clip(np.where(vals.isna().to_numpy(), rate, p), 0.0, 1.0)
        return p, {
            "mechanism": "MNAR", "mode": "random_split",
            "high_levels": [_j(u) for u in sorted(high, key=repr)],
            "n_levels": len(uniques),
            "target_rate": float(rate), "expected_rate": float(p.mean()),
            "warning": ("R0 comparison mode: reproduces B45 (shared RNG state) and B46 "
                        "(a single-level column gets every row at rate*cat_high_mult)"),
        }

    # ---- semantic groups ---------------------------------------------------
    if mode == "semantic_groups":
        if not col_spec.groups:
            raise ValueError(
                f"MNAR column {column!r}: mode 'semantic_groups' requires `groups` in "
                f"configs/missingness.yaml"
            )
        offsets = col_spec.group_offsets or {}
        eta = np.zeros(n, dtype=float)
        assigned: Dict[str, Any] = {}
        vals = pd.Series(s)
        for gname, levels in col_spec.groups.items():
            off = float(offsets.get(gname, 0.0))
            sel = vals.isin(list(levels)).to_numpy()
            eta[sel] = off
            assigned[gname] = {"levels": [_j(l) for l in levels], "offset": off,
                               "n_rows": int(sel.sum())}
        uncovered = int((~vals.isin([l for ls in col_spec.groups.values() for l in ls])).sum())
        if float(np.std(eta)) == 0.0:
            return np.full(n, float(rate)), {
                "mechanism": "MNAR", "mode": "semantic_groups", "groups": assigned,
                "uncovered_rows": uncovered, "degenerate": True,
                "target_rate": float(rate), "expected_rate": float(rate),
                "note": "all rows fell in one group -> reduces to MCAR at the target rate"}
        p, b, expected = fit_logistic_propensity(eta, rate)
        return p, {
            "mechanism": "MNAR", "mode": "semantic_groups", "groups": assigned,
            "uncovered_rows": uncovered, "intercept": b,
            "target_rate": float(rate), "expected_rate": expected,
            "p_min": float(p.min()), "p_max": float(p.max()), "degenerate": False,
        }

    # ---- default: continuous logit / ordinal logit -------------------------
    if mode not in ("logit", "ordinal"):  # pragma: no cover - blocked by validate()
        raise ValueError(f"unknown MNAR mode {mode!r} for column {column!r}")

    z, enc = encode_column(s, meta, col_spec.level_order)

    if enc.get("degenerate"):
        # B46: eICU vasopressor_use_std is constant 1.0. R0's random_split put
        # EVERY row at rate*1.5 and the table-level calibration could not undo it
        # per column, which is how a 30% target became 43.4% (B39). A constant
        # column carries no information about its own missingness, so the only
        # defensible MNAR propensity is the constant target rate: the column
        # degenerates to MCAR and this is recorded, not hidden.
        return np.full(n, float(rate)), {
            "mechanism": "MNAR", "mode": mode, **enc,
            "coefficient": 0.0, "intercept": float(np.log(rate / (1 - rate))),
            "target_rate": float(rate), "expected_rate": float(rate),
            "note": ("zero-variance column: MNAR is undefined, falls back to a constant "
                     "propensity equal to the target rate (fixes B46/B39)"),
        }

    a = col_spec.coefficient
    source = "config"
    if a is None:
        a = _auto_coefficient(rng, spec.mnar_strength, spec.mnar_min_abs)
        source = "auto"
    a = float(a)
    p, b, expected = fit_logistic_propensity(a * z, rate)
    return p, {
        "mechanism": "MNAR", "mode": mode, **enc,
        "coefficient": a, "coefficient_source": source, "intercept": b,
        "target_rate": float(rate), "expected_rate": expected,
        "p_min": float(p.min()), "p_max": float(p.max()), "degenerate": False,
    }


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def build_propensity_matrix(
    df: pd.DataFrame,
    spec: MissingnessSpec,
    registry,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Assemble the full ``(n_rows, n_cols)`` propensity matrix.

    Observed columns (identifier, declared ``always_observed``, and every MAR
    driver) get propensity 0 and are therefore never masked — the strict-MAR
    guarantee of ``missing_data_generator.py:638-654``, extended to all three
    mechanisms.
    """
    cols = spec.schema.order()
    cols = [c for c in cols if c in df.columns]
    n = len(df)
    P = np.zeros((n, len(cols)), dtype=float)
    records: Dict[str, Dict[str, Any]] = {}

    observed = set(spec.observed_columns())
    targets = [c for c in cols if c not in observed]

    legacy_row_p: Optional[np.ndarray] = None
    if spec.mechanism == "MAR" and spec.mar_mode == "row_broadcast_legacy":
        drivers = spec.driver_union()
        legacy_row_p, legacy_rec = mar_propensity_legacy_row(df, drivers, spec, spec.rate)

    for j, c in enumerate(cols):
        if c in observed:
            records[c] = {"masked": False,
                          "reason": ("MAR driver" if c in set(spec.driver_union())
                                     else "identifier/always_observed")}
            continue

        rate_c = spec.rate_for(c)

        if spec.mechanism == "MCAR":
            p, rec = mcar_propensity(n, rate_c)
        elif spec.mechanism == "MAR":
            if legacy_row_p is not None:
                p, rec = legacy_row_p.copy(), dict(legacy_rec)
            else:
                p, rec = mar_propensity(
                    df, c, spec.mar_for(c), spec, rate_c,
                    registry.stream("propensity", spec.mechanism, c),
                )
        else:  # MNAR
            p, rec = mnar_propensity(
                df, c, spec.mnar_for(c), spec, rate_c,
                registry.stream("propensity", spec.mechanism, c),
            )

        P[:, j] = p
        rec["masked"] = True
        records[c] = rec

    return P, records
