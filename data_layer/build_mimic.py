"""T2.1: build the replacement MIMIC derived table.

P2 section 1.3 adopts `~/data_MIMIC_ICU/mimic_icu_mortality_real.csv` (2939 x 31)
in place of the current 2052 x 8 table, whose `ALARM` target is indefensible (34
classes, 9 of them singletons, imbalance 792:1, stratified CV mathematically
impossible) and which has no build script anywhere on the machine (B36).

The candidate is not usable as shipped. T1.6 section 6.3.7 catalogued ten defects
(F1-F10); this script applies the column-level fixes, all of which are pure
operations on the CSV and need no database access.

Every decision below is recorded in the emitted `mimic_build_report.json` so the
resulting table is auditable.
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
SRC = Path(os.environ.get(
    "MIMIC_RAW_CSV", str(Path(__file__).resolve().parents[2]
                         / "data_MIMIC_ICU"
                         / "mimic_icu_mortality_real.csv")))

#: F1 + F2. itemid 220739 is "GCS - Eye Opening" (1-4), not the total; the
#: COALESCE at mimic_extract_v7.py:280-281 adopts it as the total and the
#: clip(3,15) at :300 folds {1,2,3} into 3, leaving only {3,4}. GCS_severe
#: (= GCS < 8) is therefore 1 for all 2939 rows. P2 section 1.3 says delete
#: rather than relabel, because we cannot return to the database to verify.
DROP_GCS = ["GCS", "GCS_severe"]

#: F3. Hardcoded constant 24 at mimic_extract_v7.py:601 - zero variance, the
#: same defect as eICU's B35.
DROP_ZERO_VARIANCE = ["hours_since_admission"]

#: F10. Deterministic functions of columns that are themselves present, all
#: verified to reconstruct 2939/2939. Keeping them would repeat exactly the
#: B34/B41 problem we are removing elsewhere.
DROP_DERIVED = ["MAP_below_65", "lactate_elevated", "lactate_severely_elevated",
                "ARDS_mild", "ARDS_moderate", "AKI_stage1", "composite_risk_score"]

#: Record keys, not features.
DROP_IDENTIFIERS = ["stay_id", "subject_id"]

#: F9. 8 rows have icu_death=1 while mortality_risk=0, and the two agree on
#: 95.3% of rows otherwise - it is a near-duplicate of the target with an
#: unexplained inconsistency. Dropping it removes both problems at once.
DROP_TARGET_LEAK = ["icu_death"]

#: B68, found in P2 and not among T1.6's F1-F10: vasopressor_use ==
#: (vasopressor_dose > 0) holds for 2939/2939 rows exactly. Another B41-type
#: deterministic redundancy. The precondition check below re-verifies it at every
#: build rather than trusting this comment.
DROP_REDUNDANT = ["vasopressor_use"]

#: Downstream target. Per the Q1-4 principle a target is never an imputation
#: feature; it is carried in the table for the downstream task only.
TARGET = "mortality_risk"

CATEGORICAL = ["gender", "mechanical_ventilation"]

#: F4. The Charlson comorbidity index has a clinical ceiling near 37; the SQL at
#: :511-518 sums over every admission and every diagnosis row for a patient, so
#: 9.29% of rows exceed it, up to 356. Winsorized rather than dropped: the
#: ordering still carries signal, only the scale is corrupted.
CHARLSON_CAP = 37

#: F5, refined by B69. On a log scale the non-zero doses are
#: clearly bimodal: a main mode over 10^-1 to 10^1 (norepinephrine equivalent in
#: mcg/kg/min, physiologically sensible), a near-empty gap at 10^1.5-10^2 (4
#: values), then a second mode of 197 values (15.3%) at 10^2-10^3. Two units are
#: mixed in one column. We cannot recover which row is which without the
#: database, so the high mode is winsorized to a defensible ceiling and the
#: problem is disclosed rather than silently smoothed.
VASO_CAP = 10.0


def build(out_csv: Path, out_report: Path) -> dict:
    raw = pd.read_csv(SRC)
    rep: dict = {"source": str(SRC), "source_shape": list(raw.shape), "steps": []}

    def note(step: str, **kw):
        rep["steps"].append({"step": step, **kw})

    # --- verify the claims this script relies on, rather than trusting them ---
    checks = {}
    checks["vasopressor_use_is_deterministic"] = bool(
        (raw.vasopressor_use == (raw.vasopressor_dose > 0).astype(int)).all())
    checks["GCS_only_two_levels"] = sorted(raw.GCS.unique().tolist())
    checks["GCS_severe_zero_variance"] = int(raw.GCS_severe.nunique())
    checks["hours_since_admission_constant"] = int(raw.hours_since_admission.nunique())
    checks["icu_death_contradictions"] = int(((raw.icu_death == 1) & (raw[TARGET] == 0)).sum())
    checks["charlson_above_clinical_max"] = int((raw.charlson_index > CHARLSON_CAP).sum())
    checks["vaso_dose_above_cap"] = int((raw.vasopressor_dose > VASO_CAP).sum())
    rep["preconditions"] = checks
    if not checks["vasopressor_use_is_deterministic"]:
        raise SystemExit("precondition failed: vasopressor_use is not a function "
                         "of vasopressor_dose; re-examine before dropping it")

    df = raw.copy()

    for group, cols, why in [
        ("GCS (F1/F2)", DROP_GCS, "itemid 220739 is Eye Opening, not the total; "
                                  "GCS has only {3,4} and GCS_severe is constant 1"),
        ("zero variance (F3)", DROP_ZERO_VARIANCE, "hardcoded constant 24"),
        # Precise statement: these reconstruct exactly from EACH OTHER, not from
        # the physiology that survives. composite_risk_score is the sum of the
        # indicator columns beside it -- with them present it fits at R^2 = 1.0,
        # with them gone only 0.485. They are dropped as a group for that reason.
        # (eICU's like-named column is NOT dropped: it reconstructs at only
        # R^2 = 0.671 there and is that dataset's downstream target.)
        ("deterministic derived (F10)", DROP_DERIVED,
         "exact linear combinations of one another; the group is redundant as a "
         "block, and MIMIC already has a better target in mortality_risk"),
        ("identifiers", DROP_IDENTIFIERS, "record keys, not features"),
        ("target leak (F9)", DROP_TARGET_LEAK, "near-duplicate of the target, 8 rows contradict it"),
        ("redundant (new)", DROP_REDUNDANT, "vasopressor_use == (vasopressor_dose>0) exactly"),
    ]:
        present = [c for c in cols if c in df.columns]
        df = df.drop(columns=present)
        note("drop", group=group, columns=present, reason=why)

    # --- winsorization, both disclosed, neither destructive ---
    #
    # Unlike eICU's urine_output (which keeps its raw values because Q1-19 asks
    # for raw/clip/log all to be reported), these two MUST be clipped in the
    # shipped table: P2 section 4.4 criterion 4 requires every continuous column
    # to lie within physiological range, and an unclipped charlson_index of 356
    # or a vasopressor_dose of 2660 would trigger the 2026-08-06 fallback gate.
    #
    # So the values are clipped here AND the originals are written to a sidecar,
    # which keeps the same three-convention sensitivity analysis available for
    # MIMIC without weakening the gate.
    # The raw values ride along as temporary columns so they survive the
    # complete-case step and can be aligned with the final IDs. Building the
    # sidecar here instead would number its rows against the pre-dropna table
    # and silently mis-align it with the shipped one.
    df["_charlson_index_raw"] = df.charlson_index
    df["_vasopressor_dose_raw"] = df.vasopressor_dose

    n_ch = int((df.charlson_index > CHARLSON_CAP).sum())
    ch_max = float(df.charlson_index.max())
    df["charlson_index"] = df.charlson_index.clip(upper=CHARLSON_CAP)
    note("winsorise", column="charlson_index", cap=CHARLSON_CAP, rows_affected=n_ch,
         max_before=ch_max, raw_values_preserved_in="MIMIC_preclip_values.csv",
         reason="SQL double-counts across admissions and diagnosis rows (F4)")

    n_vd = int((df.vasopressor_dose > VASO_CAP).sum())
    vd_max = float(df.vasopressor_dose.max())
    df["vasopressor_dose"] = df.vasopressor_dose.clip(upper=VASO_CAP)
    note("winsorise", column="vasopressor_dose", cap=VASO_CAP, rows_affected=n_vd,
         max_before=vd_max, raw_values_preserved_in="MIMIC_preclip_values.csv",
         reason="two units mixed in one column, bimodal on a log scale (F5/B69)")

    # --- complete case ---
    before = len(df)
    na_by_col = {k: int(v) for k, v in df.isna().sum().items() if v}
    df = df.dropna().reset_index(drop=True)
    note("complete_case", rows_before=before, rows_after=len(df),
         retention=round(len(df) / before, 4), missing_by_column=na_by_col,
         reason="the benchmark masks a complete table; retention is 96.9%, "
                "unlike eICU's 28.6%")

    # --- encode gender, add ID ---
    df["gender"] = (df.gender == "M").astype(int)
    note("encode", column="gender", mapping={"M": 1, "F": 0})

    df.insert(0, "ID", range(1, len(df) + 1))

    # Split the raw values off now that every surviving row has its final ID.
    preclip = df[["ID", "_charlson_index_raw", "_vasopressor_dose_raw"]].rename(
        columns={"_charlson_index_raw": "charlson_index_raw",
                 "_vasopressor_dose_raw": "vasopressor_dose_raw"})
    df = df.drop(columns=["_charlson_index_raw", "_vasopressor_dose_raw"])

    features = [c for c in df.columns if c not in ("ID", TARGET)]
    rep["final"] = {
        "shape": list(df.shape),
        "n_features": len(features),
        "target": TARGET,
        "categorical": [c for c in CATEGORICAL if c in features],
        "continuous": [c for c in features if c not in CATEGORICAL],
        "target_balance": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    preclip.to_csv(out_csv.parent / "MIMIC_preclip_values.csv", index=False)
    out_report.write_text(json.dumps(rep, indent=2))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", default=str(root / "data" / "derived" / "MIMIC_complete.csv"))
    ap.add_argument("--report", default=str(root / "data" / "derived" / "mimic_build_report.json"))
    a = ap.parse_args()
    rep = build(Path(a.out), Path(a.report))
    print(json.dumps({"preconditions": rep["preconditions"], "final": rep["final"]},
                     indent=2, ensure_ascii=False))
    for s in rep["steps"]:
        cols = s.get("columns", [s.get("column")])
        print(f"  {s['step']:<14} {str(cols):<70} {s.get('rows_affected', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
