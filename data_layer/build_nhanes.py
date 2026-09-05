"""T2.1: rebuild the NHANES derived table with skip-aware gating variables.

Two problems are fixed here.

**The gating variables were missing entirely.** `fm_NHANES_toCSV_forSNI_v1.py`
asks for `BPQ_J`, `DIQ_J`, `SMQ_J` and `FASTQX_J` (lines 50-53) but those four
XPT modules were never downloaded, and the script degrades silently (B61): it
emits 11 features instead of 16 without an error. The five variables lost that
way -- `bp_med_std`, `lipid_med_std`, `glucose_med_std`, `smoking_std`,
`fasting_state_std` -- are exactly the ones that make a defensible MAR mechanism.

**The derivation treats structural zeros as missing.** NHANES routes questions
through skip patterns: `SMQ040` ("do you smoke now") is only asked of respondents
who answered `SMQ020 == 1` ("smoked 100 cigarettes in your life"). The original
script maps anything that is not 1/2/3 to NaN (lines 176-198), so all 1329
never-smokers become "missing" when they are, by definition, non-smokers.

Applying the original logic to the completed modules would drop n from 2274 to
**184 (-91.9%)**. Deriving from the parent question instead costs **22 rows
(-0.97%)**. The 92% loss is an artifact of the derivation, not a property of the
data, so the skip-aware version is mandatory rather than optional.

Note also that v4.3 of the script had a `fillna(0.0)` at line 290-293 which was
semantically correct for exactly these skips; `forSNI_v1` removed it and
commented "gating features (keep NaN, do not fill)" at :288.

Additionally fixed: the lost `.fillna(0)` in the fasting calculation (B63,
`:202` vs `fm_NHANES_toCSV_v4_3.py:190`), which would silently mark a fasting
participant as non-fasting when `PHAFSTMN` is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Child item -> (parent item, parent value meaning "route around", derived value).
#: Each entry says: when the parent takes this value the child was never asked,
#: and the semantically correct derived value is 0, not missing.
SKIP_RULES = {
    "bp_med_std":      [("BPQ020", 2, 0.0), ("BPQ040A", 2, 0.0)],
    "lipid_med_std":   [("BPQ080", 2, 0.0)],
    "glucose_med_std": [("DIQ010", 2, 0.0), ("DIQ010", 3, None)],  # 3 = borderline, keep asked value
    "smoking_std":     [("SMQ020", 2, 0.0)],
}

#: Child items and their 1/2-style encodings.
CHILD_MAP = {
    "bp_med_std":      ("BPQ050A", {1: 1.0, 2: 0.0}),
    "lipid_med_std":   ("BPQ090D", {1: 1.0, 2: 0.0}),
    "glucose_med_std": ("DIQ070",  {1: 1.0, 2: 0.0}),
    "smoking_std":     ("SMQ040",  {1: 1.0, 2: 1.0, 3: 0.0}),
}

FILES = {
    "DEMO_J.XPT":    ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"],
    "BMX_J.XPT":     ["SEQN", "BMXBMI", "BMXWAIST"],
    "BPX_J.XPT":     ["SEQN", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3"],
    "HDL_J.XPT":     ["SEQN", "LBDHDD"],
    "TRIGLY_J.XPT":  ["SEQN", "LBXTR"],
    "GLU_J.XPT":     ["SEQN", "LBXGLU"],
    "GHB_J.XPT":     ["SEQN", "LBXGH"],
    # parent items included alongside the child items
    "BPQ_J.XPT":     ["SEQN", "BPQ020", "BPQ040A", "BPQ050A", "BPQ080", "BPQ090D"],
    "DIQ_J.XPT":     ["SEQN", "DIQ010", "DIQ070"],
    "SMQ_J.XPT":     ["SEQN", "SMQ020", "SMQ040"],
    "FASTQX_J.XPT":  ["SEQN", "PHAFSTHR", "PHAFSTMN"],
}

CORE = ["waist_circumference", "systolic_bp", "diastolic_bp", "triglycerides",
        "hdl_cholesterol", "fasting_glucose", "age", "bmi", "hba1c"]
GATES = ["bp_med_std", "lipid_med_std", "glucose_med_std", "smoking_std",
         "fasting_state_std"]
TARGET = "metabolic_score"


def load_modules(raw_dir: Path, report: dict) -> pd.DataFrame:
    df = None
    for fn, cols in FILES.items():
        p = raw_dir / fn
        if not p.exists():
            raise SystemExit(f"missing module {p}. Download with:\n"
                             f"  curl -sS -o {p} https://wwwn.cdc.gov/Nchs/Data/"
                             f"Nhanes/Public/2017/DataFiles/{p.stem.lower()}.xpt")
        m = pd.read_sas(p, format="xport")
        keep = [c for c in cols if c in m.columns]
        report["modules"][fn] = {"rows": len(m), "kept_columns": keep,
                                 "absent_columns": [c for c in cols if c not in m.columns]}
        m = m[keep]
        df = m if df is None else df.merge(m, on="SEQN", how="left")
    return df


def derive(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    out = pd.DataFrame({"SEQN": df.SEQN})

    out["age"] = df.RIDAGEYR
    out["gender_std"] = (df.RIAGENDR == 2).astype(float)      # 1=M, 2=F -> 0/1
    out["bmi"] = df.BMXBMI
    out["waist_circumference"] = df.BMXWAIST
    # B71. A blood-pressure reading of zero is NHANES's failed-measurement code,
    # not a measurement, and it must be removed BEFORE the three readings are
    # averaged -- otherwise one failed cuff inflation drags the mean down and
    # produces a "diastolic pressure" of 2.7 mmHg.
    #
    # Two traps here. First, `pandas.read_sas` decodes the SAS zero out of XPT's
    # IBM hexadecimal float as the denormal 5.397605346934028e-79, not as 0.0, so
    # an `== 0` test matches nothing: BPXDI1/2/3 contain zero exact zeros and
    # 134/124/127 values in (0, 1). Second, R0's script (fm_NHANES_toCSV_forSNI
    # _v1.py:138-140) simply averages the three columns with no zero handling at
    # all, which is why its shipped table has 8 rows at a diastolic pressure
    # below 1 mmHg and 11 below 20.
    # 20 mmHg, not 1. Chosen from the data rather than asserted: of 19,018 valid
    # diastolic readings only 13 (0.068 %) fall below 20 and the 0.1st percentile
    # is 24, while the affected respondents show patterns like (32, 2, 10) and
    # (0, 0, 8) -- a diastolic pressure of 2 mmHg is incompatible with life, so
    # these are failed inflations, not observations. The threshold costs nothing
    # on the systolic side, where the minimum valid reading is 72.
    BP_MIN_VALID = 20.0
    sys_cols, dia_cols = ["BPXSY1", "BPXSY2", "BPXSY3"], ["BPXDI1", "BPXDI2", "BPXDI3"]
    bp_dropped = {}
    for col in sys_cols + dia_cols:
        bad = df[col].notna() & (df[col] < BP_MIN_VALID)
        bp_dropped[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["bp_failed_readings_dropped"] = {
        "threshold_mmHg": BP_MIN_VALID, "per_column": bp_dropped,
        "note": "dropped before averaging; see finding B71",
    }

    out["systolic_bp"] = df[sys_cols].mean(axis=1)
    out["diastolic_bp"] = df[dia_cols].mean(axis=1)
    out["hdl_cholesterol"] = df.LBDHDD
    out["triglycerides"] = df.LBXTR
    out["fasting_glucose"] = df.LBXGLU
    out["hba1c"] = df.LBXGH

    # Belt and braces: if all three readings failed the row has no pressure at
    # all. (The previous version of this guard tested the *averaged* column for
    # equality with 0 and matched nothing -- see B71.)
    for col in ("systolic_bp", "diastolic_bp"):
        bad = out[col].notna() & (out[col] < BP_MIN_VALID)
        report[f"{col}_below_threshold_after_averaging"] = int(bad.sum())
        out.loc[bad, col] = np.nan

    # --- gating variables, skip-aware ---
    gate_report = {}
    for gate, (child, mapping) in CHILD_MAP.items():
        naive = df[child].map(mapping)
        v = naive.copy()
        applied = {}
        for parent, pval, derived in SKIP_RULES.get(gate, []):
            if derived is None:
                continue
            route = (df[parent] == pval) & v.isna()
            v.loc[route] = derived
            applied[f"{parent}=={pval}"] = int(route.sum())
        gate_report[gate] = {
            "child_item": child,
            "missing_naive": int(naive.isna().sum()),
            "missing_skip_aware": int(v.isna().sum()),
            "rows_recovered_by_parent": applied,
            "prevalence": float(np.nanmean(v)),
        }
        out[gate] = v

    # fasting: B63 restores the .fillna(0) that v4.3 had and forSNI_v1 dropped
    hrs = df.PHAFSTHR + df.PHAFSTMN.fillna(0) / 60.0
    out["fasting_state_std"] = (hrs >= 8).astype(float)
    out.loc[df.PHAFSTHR.isna(), "fasting_state_std"] = np.nan
    gate_report["fasting_state_std"] = {
        "child_item": "PHAFSTHR+PHAFSTMN",
        "missing_naive": int(df.PHAFSTHR.isna().sum()),
        "missing_skip_aware": int(out.fasting_state_std.isna().sum()),
        "rows_recovered_by_parent": {},
        "prevalence": float(np.nanmean(out.fasting_state_std)),
        "note": "B63: PHAFSTMN.fillna(0) restored from v4.3:190",
    }
    report["gates"] = gate_report

    # --- metabolic syndrome score: the DOWNSTREAM TARGET, never a feature ---
    male = out.gender_std == 0
    comp = pd.DataFrame({
        "waist": ((male & (out.waist_circumference >= 102)) |
                  (~male & (out.waist_circumference >= 88))).astype(float),
        "tg": (out.triglycerides >= 150).astype(float),
        "hdl": ((male & (out.hdl_cholesterol < 40)) |
                (~male & (out.hdl_cholesterol < 50))).astype(float),
        "bp": ((out.systolic_bp >= 130) | (out.diastolic_bp >= 85)).astype(float),
        "glu": (out.fasting_glucose >= 100).astype(float),
    })
    out[TARGET] = comp.sum(axis=1)

    return out[out.age >= 18].reset_index(drop=True)


def build(raw_dir: Path, out_csv: Path, out_report: Path) -> dict:
    report: dict = {"raw_dir": str(raw_dir), "modules": {}}
    merged = load_modules(raw_dir, report)
    report["merged_shape"] = list(merged.shape)

    d = derive(merged, report)
    report["adults_shape"] = list(d.shape)

    feature_cols = CORE + ["gender_std"] + GATES
    need = feature_cols + [TARGET]

    # Counterfactual: what the ORIGINAL (skip-blind) logic would have cost.
    naive_gate_na = {g: report["gates"][g]["missing_naive"] for g in GATES}
    report["counterfactual_naive"] = {
        "per_gate_missing": naive_gate_na,
        "note": "T1.6 measured n=184 under the original logic; the skip-aware "
                "derivation below is what makes the modules usable at all",
    }

    before = len(d)
    complete = d.dropna(subset=need).reset_index(drop=True)
    report["complete_case"] = {
        "rows_before": before, "rows_after": len(complete),
        "retention": round(len(complete) / before, 4),
        "missing_by_column": {k: int(v) for k, v in d[need].isna().sum().items() if v},
    }

    complete.insert(0, "ID", range(1, len(complete) + 1))
    keep = ["ID"] + feature_cols + [TARGET]
    complete = complete[keep]

    report["final"] = {
        "shape": list(complete.shape),
        "n_features": len(feature_cols),
        "target": TARGET,
        "categorical": ["gender_std"] + GATES,
        "continuous": CORE,
        "note": "metabolic_score is the downstream target and is NOT an "
                "imputation feature (Q1-4); age_band is not derived at all (B41)",
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(out_csv, index=False)
    out_report.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(root / "data" / "raw" / "nhanes"))
    ap.add_argument("--out", default=str(root / "data" / "derived" / "NHANES_complete.csv"))
    ap.add_argument("--report", default=str(root / "data" / "derived" / "nhanes_build_report.json"))
    a = ap.parse_args()
    rep = build(Path(a.raw_dir), Path(a.out), Path(a.report))
    print(json.dumps({k: rep[k] for k in ("merged_shape", "adults_shape",
                                          "complete_case", "final")},
                     indent=2, ensure_ascii=False))
    print("\ngating variables:")
    for g, r in rep["gates"].items():
        print(f"  {g:<20} naive NA {r['missing_naive']:>5} -> skip-aware "
              f"{r['missing_skip_aware']:>4}   prevalence {r['prevalence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
