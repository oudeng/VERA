"""P2 section 4.4: the sanity check that decides the MIMIC fallback gate.

If the cleaned MIMIC table does not pass these checks by **2026-08-06**, the
instruction is to revert to the original MIMIC table and state the provenance and
driver limitations honestly in the paper, rather than delay the full grid.

Six criteria. Five are properties of the table and are checked here:

1. no zero-variance column
2. no deterministic derived column
3. the target is not in the feature set
4. continuous columns lie within physiological range
5. every categorical class has at least 20 samples

The sixth -- that all nine methods run on MAR@30% with sensible metric
magnitudes, with no R^2 collapse -- needs an actual run and is driven separately
by `tests/sanity_nine_methods.py`.

Criterion 2 is checked two ways, because "derived" has two shapes. A column can
be a function of one other column (`vasopressor_use` from `vasopressor_dose`,
B68; `age_band` from `age_years`, B41), or a linear combination of several
(`composite_risk_score`, B34). Testing only the first would have missed the
second.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MIN_CLASS_COUNT = 20
#: A column reconstructible from the others to better than this is derived in
#: substance even if not exactly. Set well above ordinary collinearity.
LINEAR_R2_CUTOFF = 0.999

#: Criterion 4. Bounds are clinical plausibility limits for an ICU stay, not
#: distribution quantiles: the point is to catch a unit error (B59's urine
#: output at 220,912 mL/h) or a coding artifact, not to trim a tail.
PHYSIOLOGICAL: Dict[str, Dict[str, tuple]] = {
    "MIMIC": {
        "MAP_mmHg": (20, 180), "SBP_min": (30, 300), "DBP_min": (10, 200),
        "HR_max": (20, 300), "RespRate_max": (4, 80), "SpO2_min": (30, 100),
        "lactate_mmol_L": (0.1, 30), "creatinine_mg_dL": (0.1, 20),
        "hemoglobin_min": (2, 25), "sodium_min": (100, 180),
        "urine_output_min": (0, 1400), "age_years": (18, 120),
        "charlson_index": (0, 37), "vasopressor_dose": (0, 10),
    },
    "eICU": {
        "map_mmhg": (20, 180), "sbp_min": (30, 300), "dbp_min": (10, 200),
        "hr_max": (20, 300), "resprate_max": (4, 80), "spo2_min": (30, 100),
        "lactate_mmol_l": (0.1, 30), "creatinine_mg_dl": (0.1, 20),
        # A single row at 0.9 g/dL. Survivable hemoglobin below 2 is documented
        # in massive-haemorrhage case reports, so the bound was too tight.
        "hemoglobin_min": (0.5, 25), "sodium_min": (100, 180),
        "urine_output_min": (0, 1400), "age_years": (18, 120),
        "gcs": (3, 15), "vasopressor_dose": (0, 10),
    },
    "NHANES": {
        "age": (18, 120), "bmi": (10, 80), "waist_circumference": (40, 200),
        "systolic_bp": (60, 260), "diastolic_bp": (20, 150),
        # Hyperalphalipoproteinemia is real and reaches 150-200 mg/dL; the
        # earlier bound of 150 flagged two genuine values, so the bound was
        # wrong rather than the data.
        "hdl_cholesterol": (10, 200), "triglycerides": (10, 3000),
        "fasting_glucose": (30, 600), "hba1c": (3, 20),
    },
}


#: A source column with more distinct values than this fraction of the rows
#: determines every other column trivially -- a record key is the extreme case.
#: Reporting those as findings buries the real ones: on the uncleaned MIMIC
#: candidate, `subject_id` was "determining" age, sex and comorbidity index.
MAX_SOURCE_CARDINALITY_FRAC = 0.5


#: Bound violations that are a recorded decision rather than an oversight. Each
#: entry must name the finding or instruction that authorises it; an exception
#: without a reason is just a suppressed test.
DECLARED_EXCEPTIONS: Dict[str, Dict[str, str]] = {
    "eICU": {
        "urine_output_min":
            "B59 / Q1-19. The suspected unit error is left in place on purpose: "
            "the instruction requires raw, clipped and log conventions all to be "
            "reported, and clipping in the data layer would make two of the "
            "three unreproducible without a rebuild. The 1400 mL/h cap travels "
            "in configs/datasets.yaml for the evaluation layer to apply.",
    },
}


def _functional_dependencies(df: pd.DataFrame, cols: List[str]) -> List[dict]:
    """Columns that are an exact function of one other column.

    Two classes of trivial hit are excluded, both discovered by running this on
    the uncleaned MIMIC candidate:

    * a near-unique source (a record key) determines everything, so sources
      above `MAX_SOURCE_CARDINALITY_FRAC` of the row count are skipped;
    * a constant target is determined by everything, and is already reported by
      criterion 1, so zero-variance targets are skipped here.
    """
    n = len(df)
    usable_src = [c for c in cols
                  if df[c].nunique(dropna=False) <= MAX_SOURCE_CARDINALITY_FRAC * n]
    out = []
    for target in cols:
        if df[target].nunique(dropna=False) <= 1:
            continue
        for src in usable_src:
            if src == target:
                continue
            g = df.groupby(src, dropna=False)[target].nunique(dropna=False)
            if len(g) > 1 and (g <= 1).all():
                out.append({"column": target, "determined_by": src,
                            "n_groups": int(len(g))})
                break
    return out


def _linear_reconstructions(df: pd.DataFrame, cols: List[str]) -> List[dict]:
    """Columns reconstructible as a linear combination of the others.

    Rows with any missing value are dropped first. Passing NaNs to `lstsq` does
    not raise a Python exception -- LAPACK prints "Intel oneMKL ERROR: Parameter
    4 was incorrect on entry to DGELSD" to stderr and the routine returns
    garbage, so the first version of this function silently reported *no* linear
    reconstructions on a table that has one.
    """
    out = []
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 3:
        return out
    sub = df[num].dropna()
    if len(sub) < len(num) + 2:
        return out
    X_all = sub.to_numpy(dtype=float)
    for j, c in enumerate(num):
        if sub[c].nunique() <= 1:
            continue
        y = X_all[:, j]
        X = np.column_stack([np.delete(X_all, j, axis=1), np.ones(len(X_all))])
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            continue
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - X @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        if r2 >= LINEAR_R2_CUTOFF:
            out.append({"column": c, "r2_from_others": round(r2, 8),
                        "n_rows_used": int(len(sub))})
    return out


def check(dataset: str, csv: Path, target: str, categorical: List[str],
          identifier: str = "ID") -> dict:
    df = pd.read_csv(csv)
    features = [c for c in df.columns if c not in (identifier, target)]
    rep: dict = {"dataset": dataset, "path": str(csv),
                 "shape": list(df.shape), "n_features": len(features),
                 "criteria": {}}

    # 1 -- zero variance
    zv = [c for c in features if df[c].nunique(dropna=False) <= 1]
    rep["criteria"]["1_no_zero_variance"] = {
        "pass": not zv, "offending_columns": zv}

    # 2 -- deterministic derived
    fd = _functional_dependencies(df, features)
    lr = _linear_reconstructions(df, features)
    rep["criteria"]["2_no_deterministic_derived"] = {
        "pass": not fd and not lr,
        "functional_dependencies": fd,
        "linear_reconstructions": lr,
        "linear_r2_cutoff": LINEAR_R2_CUTOFF}

    # 3 -- target excluded from features
    rep["criteria"]["3_target_not_a_feature"] = {
        "pass": target not in features, "target": target,
        "target_present_in_table": target in df.columns,
        "note": "the target is carried in the table for the downstream task and "
                "must not appear among the imputable features"}

    # 4 -- physiological ranges
    bounds = PHYSIOLOGICAL.get(dataset, {})
    exceptions = DECLARED_EXCEPTIONS.get(dataset, {})
    viol, declared = [], []
    for c, (lo, hi) in bounds.items():
        if c not in df.columns:
            viol.append({"column": c, "issue": "declared bound but column absent"})
            continue
        n_lo = int((df[c] < lo).sum())
        n_hi = int((df[c] > hi).sum())
        if not (n_lo or n_hi):
            continue
        rec = {"column": c, "bounds": [lo, hi], "n_below": n_lo, "n_above": n_hi,
               "observed_range": [float(df[c].min()), float(df[c].max())]}
        # A violation we chose on purpose is a different object from one we did
        # not notice, and reporting them in the same list would make a
        # deliberate decision look like an oversight -- or, worse, let a real
        # oversight hide behind one.
        if c in exceptions:
            declared.append({**rec, "reason": exceptions[c]})
        else:
            viol.append(rec)
    unchecked = [c for c in features
                 if c not in bounds and c not in categorical
                 and pd.api.types.is_numeric_dtype(df[c])]
    rep["criteria"]["4_physiological_range"] = {
        "pass": not viol, "violations": viol,
        "declared_exceptions": declared,
        "columns_with_no_declared_bound": unchecked}

    # 5 -- categorical class counts
    small = []
    for c in categorical:
        if c not in df.columns:
            continue
        vc = df[c].value_counts(dropna=False)
        for lvl, n in vc.items():
            if n < MIN_CLASS_COUNT:
                small.append({"column": c, "level": str(lvl), "count": int(n)})
    rep["criteria"]["5_class_counts"] = {
        "pass": not small, "minimum": MIN_CLASS_COUNT,
        "classes_below_minimum": small}

    rep["verdict"] = ("PASS" if all(v["pass"] for v in rep["criteria"].values())
                      else "FAIL")
    rep["note"] = ("Criterion 6 (nine methods run on MAR@30% with sensible "
                   "metric magnitudes) is not a table property and is checked by "
                   "tests/sanity_nine_methods.py.")
    return rep


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["MIMIC"])
    ap.add_argument("--out", default=str(ROOT / "results" / "T2.1_datalayer"))
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    overall = {}
    for ds in a.datasets:
        blk = cfg["datasets"][ds]
        p = Path(blk["complete_path"])
        csv = p if p.is_absolute() else ROOT / p
        cats = [c for c, v in blk["columns"].items()
                if v["type"] == "categorical"]
        rep = check(ds, csv, blk["downstream_target"], cats,
                    blk.get("identifier_column", "ID"))
        overall[ds] = rep

        print("=" * 70)
        print(f"{ds}: {rep['shape'][0]} x {rep['shape'][1]}  "
              f"({rep['n_features']} features)   VERDICT: {rep['verdict']}")
        print("=" * 70)
        for k, v in rep["criteria"].items():
            print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}")
            if not v["pass"]:
                for kk, vv in v.items():
                    if kk != "pass" and vv:
                        print(f"          {kk}: {vv}")
        print()

    (outdir / "sanity_check.json").write_text(
        json.dumps(overall, indent=2, default=str))
    print(f"wrote {outdir/'sanity_check.json'}")
    return 0 if all(r["verdict"] == "PASS" for r in overall.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
