"""Mask generation, auditing and on-disk artifacts.

Pipeline for one (dataset, mechanism, rate):

1. resolve :class:`~missingness.spec.MissingnessSpec` from ``configs/missingness.yaml``
2. build a per-column propensity matrix (:mod:`missingness.propensity`)
3. draw each column's mask from its **own** Bernoulli stream (:mod:`missingness.rng`)
4. pin each column's count to its target (:mod:`missingness.calibration`)
5. write ``.csv`` / ``_mask.npy`` (uint8, 1 = missing) / ``_meta.json``
6. reload the ``.npy`` through :func:`common.masks.load_and_verify` and assert it
   against ``X_missing.isna()`` — engineering principle E4, finding B38

Step 6 is not decoration. In R0 no code path ever loaded a ``.npy``
(``common/masks.py`` documents the two consumers); the released masks were never
once checked against the released tables. Here generation *fails* if they
disagree, and the check is repeated against the CSV that actually ships.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from common import masks as common_masks
from common import runconfig

from .calibration import calibrate_column_to_count, enforce_min_missing
from .propensity import build_propensity_matrix
from .rng import StreamRegistry, stable_key
from .spec import MissingnessSpec, resolve


# ---------------------------------------------------------------------------

@dataclass
class MissingnessResult:
    """Everything one generation produces."""

    X_missing: pd.DataFrame
    mask: np.ndarray                      # uint8, 1 = missing, full column set
    spec: MissingnessSpec
    meta: Dict[str, Any]
    propensity: np.ndarray = field(repr=False, default=None)
    mask_check: Optional[common_masks.MaskCheck] = None

    @property
    def mask_bool(self) -> np.ndarray:
        return self.mask.astype(bool)

    def per_row_missing_rate(self, *, eligible_only: bool = True) -> np.ndarray:
        """Fraction of masked cells per row — the quantity R1-4 is about."""
        cols = [c for c in self.spec.schema.order() if c in self.X_missing.columns]
        m = self.mask_bool
        if eligible_only:
            keep = [i for i, c in enumerate(cols) if c in set(self.spec.target_columns())]
            if not keep:
                return np.zeros(m.shape[0])
            m = m[:, keep]
        return m.mean(axis=1)

    def row_index_correlation(self, *, eligible_only: bool = True) -> float:
        """Pearson r between per-row missing rate and row position.

        Reviewer point R1-4 in one number. R0's MAR masks used ``ID`` — a
        monotone record counter — as the sole driver, producing r between +0.67
        and +0.80 across the six datasets. A defensible mechanism has r ~ 0.
        """
        r = self.per_row_missing_rate(eligible_only=eligible_only)
        idx = np.arange(r.size, dtype=float)
        if np.std(r) < 1e-12:
            return 0.0
        return float(np.corrcoef(idx, r)[0, 1])


# ---------------------------------------------------------------------------

def _cast_for_generation(df: pd.DataFrame, spec: MissingnessSpec) -> pd.DataFrame:
    """Stabilise dtypes before masking so integers do not become ``82.0``.

    Same intent as ``missing_data_generator.cast_dataframe_for_generation``
    (:215-268), but the continuous/categorical decision comes from
    ``configs/datasets.yaml`` (E1) instead of a dtype heuristic.
    """
    out = df.copy()
    for name, meta in spec.schema.columns.items():
        if name not in out.columns:
            continue
        s = out[name]
        if meta.type == "continuous":
            out[name] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif meta.type in ("categorical", "integer_index"):
            num = pd.to_numeric(s, errors="coerce")
            nn = num.dropna()
            if len(nn) > 0 and bool(((nn - nn.round()).abs() <= 1e-8).all()):
                out[name] = num.round().astype("Int64")
            elif meta.type == "categorical":
                out[name] = s.astype("category")
    return out


def _table_calibrate(mask: np.ndarray, P: np.ndarray, eligible: np.ndarray,
                     rate: float, tolerance: float, rng: np.random.Generator) -> np.ndarray:
    """Whole-table calibration over eligible cells.

    Port of ``missing_data_generator.calibrate_mask_to_rate`` (:324-411), used
    only by ``rate_calibration: table`` (the legacy profile). It is what let
    per-column rates drift while the table total looked right — finding B39.
    """
    total = int(eligible.sum())
    if total == 0:
        return mask
    target = int(round(float(rate) * total))
    new = mask.copy()
    new[~eligible] = False
    cur = int(new[eligible].sum())
    if abs(cur / total - float(rate)) <= float(tolerance):
        return new

    if cur < target:
        cand = np.flatnonzero((~new & eligible).ravel())
        if cand.size == 0:
            return new
        p = np.clip(P.ravel()[cand], 0.0, 1.0)
        k = min(target - cur, cand.size)
        chosen = (rng.choice(cand, size=k, replace=False) if p.sum() <= 0
                  else rng.choice(cand, size=k, replace=False, p=p / p.sum()))
        new.ravel()[chosen] = True
    else:
        cand = np.flatnonzero((new & eligible).ravel())
        if cand.size == 0:
            return new
        score = 1.0 - np.clip(P.ravel()[cand], 0.0, 1.0)
        k = min(cur - target, cand.size)
        chosen = (rng.choice(cand, size=k, replace=False) if score.sum() <= 0
                  else rng.choice(cand, size=k, replace=False, p=score / score.sum()))
        new.ravel()[chosen] = False
    new[~eligible] = False
    return new


def _generate_legacy(df: pd.DataFrame, spec: MissingnessSpec) -> MissingnessResult:
    """Delegate to the verbatim R0 port (``implementation: legacy_R0``)."""
    from . import legacy as _legacy

    ct = (_legacy.LegacyColumnTypes(**spec.legacy_column_types)
          if spec.legacy_column_types
          else _legacy.column_types_from_schema(spec.schema, spec.always_observed))
    drivers = spec.driver_union() or [spec.schema.identifier_column or "ID"]
    lr = _legacy.generate_legacy(
        df, dataset=spec.dataset, mechanism=spec.mechanism, rate=spec.rate, seed=spec.seed,
        column_types=ct, mar_driver_cols=drivers, mar_logistic_scale=spec.mar_logistic_scale,
        tolerance=spec.tolerance, min_missing_per_col=spec.min_missing_per_col,
        schema=spec.schema,
    )
    cols = list(lr.X_missing.columns)
    per_col = {c: float(lr.mask[:, j].mean()) for j, c in enumerate(cols)}
    observed = set(lr.column_types.excluded)
    meta: Dict[str, Any] = {
        "schema_version": 1,
        "generator": "code_SNI/missingness/legacy.py (verbatim R0 port)",
        "spec": spec.to_dict(),
        "shape": {"n_rows": int(lr.mask.shape[0]), "n_cols": int(lr.mask.shape[1])},
        "columns": cols,
        "column_types": lr.column_types.as_dict(),
        "rates": {
            "target": float(spec.rate),
            "actual_rate_eligible": lr.meta["actual_rate_eligible"],
            "actual_rate_all": lr.meta["actual_rate_all"],
            "n_eligible_cells": int(lr.mask.shape[0] * (lr.mask.shape[1] - len(observed))),
            "n_all_cells": int(lr.mask.size),
            "per_column_missing_rate": per_col,
            "columns_outside_tolerance": {
                c: r for c, r in per_col.items()
                if c not in observed and abs(r - spec.rate) > spec.tolerance},
            "tolerance": float(spec.tolerance),
        },
        "per_column_spec": {},
        "rng": {"root_seed": int(spec.seed),
                "derivation": "R0 behavior: ONE np.random.default_rng(seed) shared by every "
                              "stage and consumed in column order (finding B45)"},
        "legacy_meta": lr.meta,
        "environment": {"code_SNI_git_commit": runconfig.git_commit()},
    }
    res = MissingnessResult(X_missing=lr.X_missing, mask=lr.mask.astype(np.uint8), spec=spec,
                            meta=meta, propensity=None)
    meta["row_index_diagnostics"] = {
        "pearson_r_rowrate_vs_rowindex_eligible": res.row_index_correlation(eligible_only=True),
        "pearson_r_rowrate_vs_rowindex_all": res.row_index_correlation(eligible_only=False),
        "interpretation": "R0 reproduction: MAR is expected to show r >> 0 (reviewer point R1-4)",
    }
    return res


def generate(
    df: pd.DataFrame,
    spec: MissingnessSpec,
    *,
    allow_input_missing: bool = False,
) -> MissingnessResult:
    """Generate one mask from a resolved spec. No files are written."""
    if not allow_input_missing and bool(df.isna().to_numpy().any()):
        raise ValueError(
            "input frame already contains missing values; ground-truth evaluation needs a "
            "complete table (pass allow_input_missing=True only for diagnostics)"
        )

    cols = [c for c in spec.schema.order() if c in df.columns]
    unknown = [c for c in df.columns if c not in spec.schema.columns]
    if unknown:
        raise ValueError(f"columns absent from configs/datasets.yaml: {unknown}")
    df = df[cols]

    if spec.implementation == "legacy_R0":
        return _generate_legacy(df, spec)

    registry = StreamRegistry(
        spec.seed, namespace=(spec.dataset, spec.mechanism, f"{spec.rate:g}", spec.profile)
    )

    # Optional row-order randomization. See MissingnessSpec.row_order_mode for
    # why this exists and why it is off by default.
    permutation: Optional[np.ndarray] = None
    if spec.row_order_mode == "shuffle":
        shuffle_seed = int(spec.row_order_seed if spec.row_order_seed is not None else spec.seed)
        permutation = np.random.default_rng(
            np.random.SeedSequence(entropy=[shuffle_seed, stable_key(spec.dataset, "row_order")])
        ).permutation(len(df))
        df = df.iloc[permutation].reset_index(drop=True)

    df_cast = _cast_for_generation(df, spec)
    n, d = df_cast.shape

    P, prop_records = build_propensity_matrix(df_cast, spec, registry)

    observed = set(spec.observed_columns())
    eligible = np.zeros((n, d), dtype=bool)
    for j, c in enumerate(cols):
        eligible[:, j] = c not in observed

    # --- draw, one private Bernoulli stream per column ---------------------
    mask = np.zeros((n, d), dtype=bool)
    for j, c in enumerate(cols):
        if c in observed:
            continue
        u = registry.stream("bernoulli", spec.mechanism, c).random(n)
        mask[:, j] = u < P[:, j]

    # --- per-column rate calibration (fixes B39) ---------------------------
    if spec.rate_calibration == "per_column":
        for j, c in enumerate(cols):
            if c in observed:
                continue
            target = max(int(round(spec.rate_for(c) * n)), int(spec.min_missing_per_col))
            mask[:, j] = calibrate_column_to_count(
                mask[:, j], P[:, j], target, registry.stream("calibration", spec.mechanism, c)
            )
    else:
        for j, c in enumerate(cols):
            if c in observed:
                continue
            mask[:, j] = enforce_min_missing(
                mask[:, j], P[:, j], int(spec.min_missing_per_col),
                registry.stream("min_per_col", spec.mechanism, c),
            )
        if spec.rate_calibration == "table":
            mask = _table_calibrate(mask, P, eligible, spec.rate, spec.tolerance,
                                    registry.stream("table", spec.mechanism, "calibration"))
            for j, c in enumerate(cols):
                if c in observed:
                    continue
                mask[:, j] = enforce_min_missing(
                    mask[:, j], P[:, j], int(spec.min_missing_per_col),
                    registry.stream("min_per_col", spec.mechanism, c),
                )

    mask[~eligible] = False  # hard guarantee: drivers and identifiers stay observed

    X_missing = df_cast.mask(pd.DataFrame(mask, index=df_cast.index, columns=cols))

    # --- bookkeeping -------------------------------------------------------
    per_col_rate = {c: float(mask[:, j].mean()) for j, c in enumerate(cols)}
    n_elig = int(eligible.sum())
    actual_eligible = float(mask[eligible].mean()) if n_elig else 0.0
    actual_all = float(mask.mean())

    for j, c in enumerate(cols):
        rec = prop_records.setdefault(c, {})
        rec["achieved_rate"] = per_col_rate[c]
        if rec.get("masked"):
            rec["target_rate"] = float(spec.rate_for(c))
            rec["abs_rate_error"] = abs(per_col_rate[c] - float(spec.rate_for(c)))
            rec["n_missing"] = int(mask[:, j].sum())

    over = {c: r for c, r in per_col_rate.items()
            if prop_records.get(c, {}).get("masked") and abs(r - spec.rate_for(c)) > spec.tolerance}

    meta: Dict[str, Any] = {
        "schema_version": 1,
        "generator": "code_SNI/missingness (T1.5 rewrite)",
        "replaces": ("project_sni_R0/sni/utility_missing_data_gen_v1/"
                     "missing_data_generator.py"),
        "spec": spec.to_dict(),
        "shape": {"n_rows": int(n), "n_cols": int(d)},
        "columns": list(cols),
        "column_types": {
            "continuous": [c for c in cols if spec.schema.columns[c].type == "continuous"],
            "categorical": [c for c in cols if spec.schema.columns[c].type == "categorical"],
            "observed": sorted(observed & set(cols)),
        },
        "rates": {
            "target": float(spec.rate),
            # both denominators, as R0 reported at missing_data_generator.py:754-759
            "actual_rate_eligible": actual_eligible,
            "actual_rate_all": actual_all,
            "n_eligible_cells": n_elig,
            "n_all_cells": int(n * d),
            "per_column_missing_rate": per_col_rate,
            "columns_outside_tolerance": over,
            "tolerance": float(spec.tolerance),
        },
        "per_column_spec": prop_records,
        "rng": registry.describe(),
        "row_order": {
            "mode": spec.row_order_mode,
            "seed": spec.row_order_seed,
            "permutation": permutation.tolist() if permutation is not None else None,
            # P2b decision 3. Shuffling is not inverted, so this mask indexes the
            # permuted table and pairing it with the unpermuted one would be
            # silently wrong -- same shapes, same dtypes, every metric computable,
            # every number meaningless. The fingerprint is of the identifier
            # column in the order the mask was built against; common/rowspace.py
            # asserts on it wherever table, mask and downstream target meet.
            "rowspace_digest": _rowspace_digest(df, spec),
        },
        "row_index_diagnostics": {},
        "environment": {
            "code_SNI_git_commit": runconfig.git_commit(),
            "libraries": {k: v for k, v in runconfig.library_versions().items()
                          if k in ("numpy", "pandas", "scipy")},
        },
    }

    res = MissingnessResult(X_missing=X_missing, mask=mask.astype(np.uint8), spec=spec,
                            meta=meta, propensity=P)
    meta["row_index_diagnostics"] = {
        "pearson_r_rowrate_vs_rowindex_eligible": res.row_index_correlation(eligible_only=True),
        "pearson_r_rowrate_vs_rowindex_all": res.row_index_correlation(eligible_only=False),
        # A non-zero r above can come from the mechanism OR from the source
        # table arriving sorted. Recording the driver-vs-row-index correlation
        # makes the two distinguishable without re-running anything.
        "pearson_r_driver_vs_rowindex": _driver_row_correlations(df_cast, spec),
        "interpretation": ("|r| < 0.05 is the T1.5 acceptance threshold and the direct "
                           "refutation of reviewer point R1-4; R0's MAR masks scored "
                           "+0.67 (MIMIC) to +0.80 (eICU). If |r| exceeds it, check "
                           "pearson_r_driver_vs_rowindex first: a driver that is also "
                           "the table's sort key transmits row order into the mask no "
                           "matter how sound the mechanism is."),
    }
    return res


def _driver_row_correlations(df: pd.DataFrame, spec: MissingnessSpec) -> Dict[str, float]:
    """corr(column, row index) for every MAR driver (and MNAR target column)."""
    cols = spec.driver_union() if spec.mechanism == "MAR" else (
        spec.target_columns() if spec.mechanism == "MNAR" else [])
    idx = np.arange(len(df), dtype=float)
    out: Dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).any() or np.nanstd(v) < 1e-12:
            out[c] = 0.0
            continue
        out[c] = float(np.corrcoef(idx, np.nan_to_num(v, nan=float(np.nanmean(v))))[0, 1])
    return out


# ---------------------------------------------------------------------------

def generate_and_write(
    df: pd.DataFrame,
    spec: MissingnessSpec,
    outdir: Path,
    *,
    stem: Optional[str] = None,
    write_csv: bool = True,
    allow_input_missing: bool = False,
) -> MissingnessResult:
    """Generate, persist, then reload and verify the mask (E4)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = _rate_tag(spec.rate)
    stem = stem or f"{spec.dataset}_{spec.mechanism}_{tag}"

    res = generate(df, spec, allow_input_missing=allow_input_missing)

    mask_path = outdir / f"{stem}_mask.npy"
    np.save(mask_path, res.mask.astype(np.uint8))

    cols = list(res.X_missing.columns)
    checks: List[Dict[str, Any]] = []

    # E4 #1: the in-memory table the caller receives
    _, chk_mem = common_masks.load_and_verify(res.X_missing, mask_path, columns=cols, strict=True)
    checks.append({"target": "in_memory", **_chk_dict(chk_mem)})

    csv_path: Optional[Path] = None
    if write_csv:
        csv_path = outdir / f"{stem}.csv"
        res.X_missing.to_csv(csv_path, index=False)
        # E4 #2: the artifact that actually ships. A CSV round-trip is where
        # NaN/dtype surprises live, and it is the file the imputers read.
        reloaded = pd.read_csv(csv_path)
        _, chk_csv = common_masks.load_and_verify(reloaded, mask_path, columns=cols, strict=True)
        checks.append({"target": "csv_roundtrip", "path": str(csv_path), **_chk_dict(chk_csv)})

    res.mask_check = chk_mem
    res.meta["artifacts"] = {
        "mask_npy": str(mask_path),
        "csv": str(csv_path) if csv_path else None,
        "mask_dtype": "uint8", "mask_encoding": "1 = missing",
    }
    res.meta["e4_mask_verification"] = {
        "principle": "E4 - the cached .npy is the authority and is asserted against the table",
        "checks": checks,
        "all_consistent": all(c["consistent"] for c in checks),
    }

    (outdir / f"{stem}_meta.json").write_text(
        json.dumps(res.meta, indent=2, ensure_ascii=False, default=str)
    )
    return res


def _chk_dict(chk: common_masks.MaskCheck) -> Dict[str, Any]:
    return {
        "mask_path": chk.mask_path,
        "n_rows": chk.n_rows, "n_cols": chk.n_cols,
        "n_missing_mask": chk.n_missing_mask, "n_missing_isna": chk.n_missing_isna,
        "n_disagreements": chk.n_disagreements, "consistent": chk.consistent,
    }



def _rowspace_digest(df, spec) -> Optional[str]:
    """Fingerprint of the row ordering the mask was built against (P2b decision 3).

    Taken from the identifier column after any shuffle, so it identifies the
    exact row space this mask indexes. Returns None when the frame carries no
    identifier -- `common.rowspace` then reports the pairing as unverifiable
    rather than treating it as verified.
    """
    from common.rowspace import fingerprint
    ident = getattr(spec.schema, "identifier_column", None) if spec.schema else None
    if not ident or ident not in df.columns:
        return None
    return fingerprint(df[ident].tolist())


def _rate_tag(rate: float) -> str:
    """``0.3 -> '30per'`` — the R0 filename convention (``format_rate``, :66-89)."""
    return f"{int(round(float(rate) * 100)):02d}per"


def generate_from_config(
    df: pd.DataFrame,
    dataset: str,
    mechanism: str,
    rate: float,
    *,
    profile: str,
    outdir: Optional[Path] = None,
    seed: Optional[int] = None,
    config_path: Optional[Path] = None,
    datasets_config_path: Optional[Path] = None,
    schema=None,
    overrides: Optional[Mapping[str, Any]] = None,
    **kw,
) -> MissingnessResult:
    """Convenience wrapper: resolve the config, then generate (and optionally write)."""
    spec = resolve(dataset, mechanism, rate, profile=profile, config_path=config_path,
                   datasets_config_path=datasets_config_path, schema=schema, seed=seed,
                   overrides=overrides)
    if outdir is None:
        return generate(df, spec, **kw)
    return generate_and_write(df, spec, Path(outdir), **kw)
