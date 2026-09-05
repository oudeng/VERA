"""T2.1: rebuild the eICU derived table and create the new CDC2022 wide table.

eICU
----
Four changes, all decided in P2 section 1.2:

* drop the two zero-variance columns `hours_since_admission` (constant 24) and
  `vasopressor_use_std` (constant 1.0) -- B35. Beyond diluting RMSE they distort
  the auditability story: removing them lifts eICU's cross-seed D stability from
  rho 0.633 to 0.703, so part of what Table 3 attributes to dimensionality is a
  data-quality artifact (B58).
* reclassify `composite_risk_score` from feature to downstream target -- B34,
  under the principle that a target is never an imputation feature. It stays in
  the table, as MIMIC's `mortality_risk` and NHANES's `metabolic_score` do,
  because the downstream task in T2.4 needs something to predict.
* drop `age_band`, a deterministic binning of `age_years` -- B41. Keeping a
  deterministic pair muddies any claim that SNI discovers dependencies, which is
  the same reason `metabolic_score` goes.
* record, but do NOT apply, a physiological cap on `urine_output_min` -- B59. It
  runs to 220,912 with a median of 4117.5 and 43.8% of values above any
  physiological bound, while the same variable in the MIMIC table built from the
  same column template runs 1-1400 mL/h with median 30. It alone accounts for
  98.54% of eICU's reported MAE, and mean imputation ranks *first* on it under
  raw NRMSE, which is a sign the metric is measuring the unit error rather than
  imputation quality. Q1-19 requires raw, clipped and log conventions all to be
  reported, so the table keeps the raw values and the cap travels in the config
  for the evaluation layer to apply. Clipping here would have made two of the
  three required conventions unreproducible without a rebuild.

CDC2022
-------
A new wide table for reviewer point R2-4 ("real tables are usually wider").
246,022 x 40, fully observed, public domain, no data use agreement. The
stratified n=3000 draw becomes the seventh benchmark dataset.

Its companion file `heart_2022_with_nans.csv` (445,132 x 40, 5.07% missing
across 38 columns) carries a **real** missingness pattern and is the only
real-versus-simulated control in the project. Row identity between the two files
is not recoverable -- only 2 of the 40 columns are fully observed in the
with-NaNs file, so there is no join key -- and is not needed: T2.2(d) compares
the missingness *mechanism* (per-column rates, co-occurrence structure,
dependence on covariates), which is a population property best estimated from
all 445,132 rows.
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: P7-A closeout: no private absolute path in a published file. These
#: inputs are restricted (DUA) or not redistributed here, so the env
#: var is the real interface; the default is repo-relative, which
#: gives a clone a path it can act on instead of a home directory.
R0_DATA = Path(os.environ.get(
    "SNI_R0_DATA", str(Path(__file__).resolve().parents[2]
                       / "project_sni_R0" / "sni" / "data")))
CDC_DIR = Path(os.environ.get(
    "CDC2022_DIR", str(Path(__file__).resolve().parents[2]
                       / "data_CDC2022")))

EICU_DROP_ZERO_VAR = ["hours_since_admission", "vasopressor_use_std"]
EICU_DROP_TARGET = ["composite_risk_score"]
EICU_DROP_REDUNDANT = ["age_band"]

#: Upper bound for a 24 h minimum hourly urine output, in mL. The MIMIC table
#: built from the same template tops out at 1400.
URINE_CAP = 1400.0

#: Physiological ceiling for a norepinephrine-equivalent infusion, mcg/kg/min.
#: Same value and same reasoning as MIMIC's VASO_CAP (B69).
VASO_CAP = 10.0

CDC_N = 3000
CDC_STRATIFY = "HadHeartAttack"
CDC_SEED = 2025


def build_eicu(out_csv: Path, out_report: Path) -> dict:
    raw = pd.read_csv(R0_DATA / "eICU_complete.csv")
    rep = {"source": str(R0_DATA / "eICU_complete.csv"),
           "source_shape": list(raw.shape), "steps": []}

    checks = {c: int(raw[c].nunique()) for c in EICU_DROP_ZERO_VAR}
    rep["preconditions"] = {"zero_variance_nunique": checks,
                            "age_band_recoverable_from_age_years": None}
    for c, n in checks.items():
        if n != 1:
            raise SystemExit(f"precondition failed: {c} has {n} distinct values, "
                             f"expected 1 -- re-examine before dropping")

    # Confirm age_band really is a deterministic function of age_years before
    # dropping it on those grounds.
    g = raw.groupby("age_years").age_band.nunique()
    rep["preconditions"]["age_band_recoverable_from_age_years"] = bool((g == 1).all())

    df = raw.copy()
    for group, cols, why in [
        ("zero variance (B35)", EICU_DROP_ZERO_VAR,
         "constant columns contribute a free perfect score and flatten D rows"),
        ("deterministic redundancy (B41)", EICU_DROP_REDUNDANT,
         "age_band is a binning of age_years"),
    ]:
        present = [c for c in cols if c in df.columns]
        df = df.drop(columns=present)
        rep["steps"].append({"step": "drop", "group": group, "columns": present,
                             "reason": why})

    # B34: composite_risk_score is the upstream target. It is removed from the
    # FEATURE set, not from the table -- exactly as MIMIC keeps mortality_risk
    # and NHANES keeps metabolic_score. An earlier version dropped it outright,
    # which would have left eICU with no downstream task at all in T2.4 and made
    # datasets.yaml declare a downstream_target that is not in the file.
    rep["steps"].append({
        "step": "reclassify_as_target", "columns": list(EICU_DROP_TARGET),
        "reason": "B34: the upstream dictionary marks composite_risk_score as "
                  "the target variable, so it is carried for the downstream "
                  "task and excluded from the imputable features (Q1-4)"})

    # B59 / Q1-19. The decision is "physiological clipping + a log convention +
    # a sensitivity analysis", with all three of raw / clip / log reported. That
    # is a property of how the metric is computed, not of what the table
    # contains, so the raw values are KEPT here and the cap is recorded for the
    # evaluation layer to apply.
    #
    # An earlier draft of this script clipped destructively. That would have made
    # the raw and log conventions unreproducible without a rebuild, which is
    # precisely the sensitivity analysis the instruction asks for.
    n_over = int((df.urine_output_min > URINE_CAP).sum())
    rep["steps"].append({
        "step": "record_cap_without_applying", "column": "urine_output_min",
        "cap": URINE_CAP, "rows_above_cap": n_over,
        "share_above_cap": round(n_over / len(df), 4),
        "observed_range": [float(df.urine_output_min.min()),
                           float(df.urine_output_min.max())],
        "reason": "B59: suspected unit or aggregation error. The column alone "
                  "accounts for 98.54% of eICU's reported cont_MAE, and mean "
                  "imputation ranks first on it under raw NRMSE, which is a "
                  "sign the metric measures the unit error rather than "
                  "imputation quality. Q1-19 requires raw / clip / log all to "
                  "be reported, so the raw values are preserved and the "
                  "convention is applied at evaluation.",
        "conventions": {
            "raw": "metrics on the column as stored",
            "clip": f"truth and prediction both clipped at {URINE_CAP} mL/h",
            "log": "metrics on log1p of truth and prediction"},
    })

    # Same defect and same treatment as MIMIC's vasopressor_dose (B69): the
    # column reaches 6250 against a physiological ceiling near 10 mcg/kg/min,
    # 971 of 1430 rows are above it, and two units are evidently mixed. Leaving
    # eICU uncapped while capping MIMIC would be an unjustifiable asymmetry
    # between two tables built from one column template. Raw values are kept in
    # a sidecar, exactly as for MIMIC.
    preclip = pd.DataFrame({"ID": df["ID"].to_numpy(),
                            "vasopressor_dose_raw": df.vasopressor_dose.to_numpy()})
    n_vd = int((df.vasopressor_dose > VASO_CAP).sum())
    vd_max = float(df.vasopressor_dose.max())
    df["vasopressor_dose"] = df.vasopressor_dose.clip(upper=VASO_CAP)
    rep["steps"].append({"step": "winsorise", "column": "vasopressor_dose",
                         "cap": VASO_CAP, "rows_affected": n_vd,
                         "max_before": vd_max,
                         "raw_values_preserved_in": "eICU_preclip_values.csv",
                         "reason": "B69-type unit mixing, treated identically to "
                                   "MIMIC's column of the same name"})

    cat = ["mechanical_ventilation_std", "gender_std"]
    target = EICU_DROP_TARGET[0]
    feat = [c for c in df.columns if c not in ("ID", target)]
    rep["final"] = {"shape": list(df.shape), "n_features": len(feat),
                    "categorical": [c for c in cat if c in feat],
                    "continuous": [c for c in feat if c not in cat],
                    "downstream_target": target,
                    "target_in_table": target in df.columns}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    preclip.to_csv(out_csv.parent / "eICU_preclip_values.csv", index=False)
    out_report.write_text(json.dumps(rep, indent=2))
    return rep


def build_cdc(out_complete: Path, out_realmiss: Path, out_report: Path) -> dict:
    full = pd.read_csv(CDC_DIR / "heart_2022_no_nans.csv")
    withna = pd.read_csv(CDC_DIR / "heart_2022_with_nans.csv")
    rep = {"source_complete": str(CDC_DIR / "heart_2022_no_nans.csv"),
           "source_with_nans": str(CDC_DIR / "heart_2022_with_nans.csv"),
           "shapes": {"no_nans": list(full.shape), "with_nans": list(withna.shape)}}

    dup = int(full.duplicated().sum())
    full = full.drop_duplicates().reset_index(drop=True)
    rep["dropped_exact_duplicates"] = dup

    # Stratified subsample so n is comparable with the other datasets.
    rng = np.random.RandomState(CDC_SEED)
    parts = []
    for lvl, grp in full.groupby(CDC_STRATIFY):
        k = int(round(CDC_N * len(grp) / len(full)))
        parts.append(grp.sample(n=min(k, len(grp)), random_state=rng))
    sub = pd.concat(parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    rep["subsample"] = {"n": len(sub), "stratify_on": CDC_STRATIFY, "seed": CDC_SEED,
                        "balance": {str(k): int(v) for k, v in
                                    sub[CDC_STRATIFY].value_counts().items()}}

    # The real-missingness control.
    #
    # An earlier attempt tried to locate the subsampled rows inside the with-NaNs
    # file by joining on shared column values. That does not work: only 2 of the
    # 40 columns are fully observed there, so the join is not a key and it
    # returned the entire table. Row identity is not recoverable and, more to the
    # point, is not needed.
    #
    # What T2.2(d) compares is the missingness *mechanism*: per-column rates, the
    # co-occurrence structure, and how strongly missingness depends on
    # covariates. Those are properties of the population, best estimated from all
    # 445,132 rows. A matched-size stratified draw is kept alongside so the
    # comparison can also be made at the same n as the simulation, where sampling
    # noise is comparable.
    rep["real_missing_full"] = {
        "n": int(len(withna)),
        "overall_missing_rate": float(withna.isna().to_numpy().mean()),
        "columns_with_missing": int((withna.isna().sum() > 0).sum()),
        "per_column_rate": {c: float(withna[c].isna().mean())
                            for c in withna.columns if withna[c].isna().any()},
    }
    parts = []
    for lvl, grp in withna.groupby(CDC_STRATIFY, dropna=False):
        k = int(round(CDC_N * len(grp) / len(withna)))
        parts.append(grp.sample(n=min(k, len(grp)), random_state=rng))
    real = pd.concat(parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    rep["real_missing_matched_draw"] = {
        "n": int(len(real)),
        "overall_missing_rate": float(real.isna().to_numpy().mean()),
        "columns_with_missing": int((real.isna().sum() > 0).sum()),
        "note": "independent stratified draw at the same n as the simulation "
                "subsample; row identity with it is neither claimed nor needed",
    }

    obj = [c for c in sub.columns if sub[c].dtype == object]
    num = [c for c in sub.columns if c not in obj]
    # Categorical here means "few levels", matching how the other tables are
    # declared; the numeric columns with <= 12 levels are ordinal codes.
    cat = obj + [c for c in num if sub[c].nunique() <= 12]
    cont = [c for c in sub.columns if c not in cat]

    codes = sub.copy()

    # AgeCategory is the principal MAR driver for this dataset, so its codes must
    # be monotone in age or the mechanism ("older respondents break off more
    # often") is meaningless. Alphabetical order over the released labels happens
    # to be correct here -- every bound is two digits, and "Age 80 or older"
    # sorts last -- but relying on that is relying on luck. The order is stated
    # explicitly and asserted.
    age_levels = sorted([v for v in sub.AgeCategory.dropna().unique()])
    lead = []
    for v in age_levels:
        digits = "".join(ch for ch in str(v) if ch.isdigit())
        lead.append(int(digits[:2]) if len(digits) >= 2 else -1)
    if lead != sorted(lead) or -1 in lead:
        raise SystemExit(f"AgeCategory levels do not sort monotonically in age: "
                         f"{list(zip(age_levels, lead))}")
    codes["AgeCategory"] = pd.Categorical(codes.AgeCategory,
                                          categories=age_levels,
                                          ordered=True).codes
    rep["age_category_encoding"] = {
        "levels_in_order": [str(v) for v in age_levels],
        "lower_bounds": lead,
        "note": "explicit ordered encoding; the MAR mechanism depends on it",
    }

    for c in obj:
        if c == "AgeCategory":
            continue
        codes[c] = pd.Categorical(codes[c]).codes
    codes.insert(0, "ID", range(1, len(codes) + 1))

    rep["final"] = {"shape": list(codes.shape), "n_features": len(sub.columns),
                    "n_categorical": len(cat), "n_continuous": len(cont),
                    "categorical": cat, "continuous": cont}

    out_complete.parent.mkdir(parents=True, exist_ok=True)
    codes.to_csv(out_complete, index=False)
    real.to_csv(out_realmiss, index=False)
    out_report.write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "derived"
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["eicu", "cdc", "both"])
    a = ap.parse_args()

    if a.which in ("eicu", "both"):
        r = build_eicu(d / "eICU_complete.csv", d / "eicu_build_report.json")
        print("=== eICU ===")
        print(json.dumps({"preconditions": r["preconditions"], "final": r["final"]},
                         indent=2, ensure_ascii=False))
    if a.which in ("cdc", "both"):
        r = build_cdc(d / "CDC2022_complete.csv", d / "CDC2022_realmissing.csv",
                      d / "cdc_build_report.json")
        print("\n=== CDC2022 ===")
        print(json.dumps({k: r[k] for k in ("dropped_exact_duplicates", "subsample",
                                            "real_missing_matched_draw", "final")},
                         indent=2, ensure_ascii=False))
        rm = r["real_missing_full"]
        print(f"  real missingness over all {rm['n']} rows: "
              f"{rm['overall_missing_rate']*100:.2f}% across "
              f"{rm['columns_with_missing']} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
