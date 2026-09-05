"""T2.1: regenerate the `datasets:` section of configs/datasets.yaml from the
derived tables that actually exist on disk.

Engineering principle E1 says configs/datasets.yaml is the single source of truth
for schema. That only holds if the file describes the tables as built rather than
as remembered, so this script derives every `cardinality`, `range` and `dtype`
from the CSV itself and takes only the *decisions* -- which column is the target,
which are categorical, which are always-observed MAR drivers -- from the build
reports and from the table below.

Everything above the `datasets:` line is preserved verbatim, because pyyaml
cannot round-trip comments and that header carries the seed policy and several
explanatory notes that must not be silently reformatted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
CONFIG = ROOT / "configs" / "datasets.yaml"

#: Per-dataset decisions. Column *facts* come from the CSV; only judgments live
#: here, each traceable to a finding or to the P2 instruction.
DECISIONS: Dict[str, dict] = {
    "MIMIC": {
        "csv": DERIVED / "MIMIC_complete.csv",
        "report": DERIVED / "mimic_build_report.json",
        "identifier": "ID",
        "target": "mortality_risk",
        # mortality_risk is binary (0/1, 2513:426, imbalance 5.9:1) so it is
        # declared categorical and the downstream task is classification. This is
        # the whole point of the replacement: the old ALARM target had 34 classes,
        # 9 of them singletons and an imbalance of 792:1, which made stratified
        # cross-validation mathematically impossible.
        "categorical": ["gender", "mechanical_ventilation", "mortality_risk"],
        "always_observed": ["age_years", "gender", "charlson_index",
                            "mechanical_ventilation"],
        "provenance": (
            "REPLACED in R1/P2. The R0 table (2052 x 9, target ALARM) had no "
            "build script anywhere on the machine (B36) and its target had 34 "
            "classes with 9 singletons and a 792:1 imbalance, so stratified CV "
            "was mathematically impossible. Rebuilt by data_layer/build_mimic.py "
            "from ~/data_MIMIC_ICU/mimic_icu_mortality_real.csv (2939 x 31), "
            "applying T1.6's F1-F10 plus B68/B69."),
    },
    "eICU": {
        "csv": DERIVED / "eICU_complete.csv",
        "report": DERIVED / "eicu_build_report.json",
        "identifier": "ID",
        "target": "composite_risk_score",
        # 14 ordered integer levels (2-15). Declared continuous, so the
        # downstream task is regression -- the same ambiguity B42 records for
        # NHANES's metabolic_score, and resolved the same way for consistency.
        "categorical": ["gender_std", "mechanical_ventilation_std"],
        "always_observed": ["age_years", "gender_std", "gcs",
                            "mechanical_ventilation_std"],
        "provenance": (
            "Rebuilt by data_layer/build_eicu_cdc.py: dropped two zero-variance "
            "columns (B35) and the deterministic age_band (B41); reclassified "
            "composite_risk_score from feature to downstream target (B34). "
            "urine_output_min is NOT winsorized in the table -- Q1-19 requires "
            "raw, clipped and log conventions all to be reported, so the 1400 "
            "mL/h cap travels in this config for the evaluation layer to apply "
            "(B59)."),
    },
    "NHANES": {
        "csv": DERIVED / "NHANES_complete.csv",
        "report": DERIVED / "nhanes_build_report.json",
        "identifier": "ID",
        "target": "metabolic_score",
        "categorical": ["gender_std", "bp_med_std", "lipid_med_std",
                        "glucose_med_std", "smoking_std", "fasting_state_std"],
        # P2b decision 2(b): fasting_state_std dropped as a driver (B72) -- the
        # complete-case step leaves it 95.9 % constant, so it cannot drive a
        # mechanism. It stays an ordinary maskable feature, which lifts NHANES's
        # evaluation coverage from 11/15 to 12/15.
        "always_observed": ["age", "gender_std", "smoking_std"],
        "provenance": (
            "Rebuilt by data_layer/build_nhanes.py after downloading the four "
            "XPT modules the original script asked for but never had "
            "(BPQ_J/DIQ_J/SMQ_J/FASTQX_J), which it degraded silently without "
            "(B61). Gating variables are derived skip-aware: applying the "
            "original skip-blind logic to the completed modules would have cut n "
            "from 2274 to 184. B63's lost .fillna(0) is restored. "
            "metabolic_score is the downstream target and is not a feature."),
    },
    "CDC2022": {
        "csv": DERIVED / "CDC2022_complete.csv",
        "report": DERIVED / "cdc_build_report.json",
        "identifier": "ID",
        "target": "HadHeartAttack",
        "categorical": None,   # inferred: object columns plus <=12 distinct codes
        "always_observed": ["Sex", "AgeCategory"],
        "provenance": (
            "NEW in R1, answering R2-4 (real tables are wider than 8-16 "
            "columns). Stratified n=3000 draw from the CDC 2022 BRFSS heart "
            "disease table (246,022 x 40 after de-duplication), public domain, "
            "no data use agreement. Its companion heart_2022_with_nans.csv "
            "carries a REAL missingness pattern and is the project's only "
            "real-versus-simulated control."),
    },
    "AutoMPG": {
        "csv": DERIVED / "AutoMPG_complete.csv",
        "report": None,
        "identifier": "ID",
        "target": "mpg",
        "categorical": ["origin", "cylinders"],
        "always_observed": ["model_year", "origin"],
        "provenance": (
            "Unchanged from R0. NOTE (B51): the table is sorted by model_year, "
            "which is also its MAR driver, so masks built on it correlate with "
            "the row index by construction. Item 2 of the T2.2(c) diagnostic "
            "attributes that to the table rather than to the mechanism."),
    },
    "ComCri": {
        "csv": DERIVED / "ComCri_complete.csv",
        "report": None,
        "identifier": "ID",
        "target": "ViolentCrimesPerPop",
        "categorical": ["IncomeLevel", "UrbanType", "EducationLevel",
                        "CrimeLevel", "RegionCode"],
        "always_observed": ["medIncome", "RegionCode", "UrbanType"],
        "provenance": "Unchanged from R0.",
    },
    "Concrete": {
        "csv": DERIVED / "Concrete_complete.csv",
        "report": None,
        "identifier": "ID",
        "target": "ConcreteCS",
        # P2 T2.1: Duration moves from categorical to continuous. It is curing
        # age in days, spanning 1-365 over 14 observed levels; treating a
        # 365-day cure as an unordered category discards the ordering the whole
        # dataset is about. This leaves Concrete with NO categorical column,
        # which is why its Accuracy/Macro-F1 cells are empty and why the Overall
        # column of Table 1 is a weighted rather than a simple mean.
        "categorical": [],
        "always_observed": ["Duration"],
        "provenance": (
            "Unchanged from R0 except that Duration is reclassified from "
            "categorical to continuous (P2 T2.1). Concrete consequently has no "
            "categorical column at all."),
    },
}


def _column_block(s: pd.Series, role: str, kind: str) -> dict:
    out = {"type": kind, "role": role, "dtype": str(s.dtype),
           "cardinality": int(s.nunique())}
    if kind != "identifier" and pd.api.types.is_numeric_dtype(s):
        out["range"] = [float(s.min()), float(s.max())]
    return out


def build_block(name: str, dec: dict) -> dict:
    df = pd.read_csv(dec["csv"])
    ident, target = dec["identifier"], dec["target"]

    cat = dec["categorical"]
    if cat is None:  # CDC2022: infer, then record what was inferred
        obj = [c for c in df.columns if df[c].dtype == object]
        cat = obj + [c for c in df.columns
                     if c not in obj and c not in (ident,) and df[c].nunique() <= 12]

    cols = {}
    for c in df.columns:
        if c == ident:
            role, kind = "identifier", "integer_index"
        elif c == target:
            role = "downstream_target"
            kind = "categorical" if c in cat else "continuous"
        else:
            # Drivers keep role "imputable". They ARE features the imputer sees;
            # they simply are never masked, and `always_observed` below is what
            # records that. An earlier version invented the role
            # "imputable_but_always_observed", which `DataSchema.from_yaml`
            # (baselines/schema.py:129) silently skips along with every other
            # non-"imputable" role -- so MIMIC would have gone into the full grid
            # with 12 features instead of 16 and nothing would have said so.
            role = "imputable"
            kind = "categorical" if c in cat else "continuous"
        cols[c] = _column_block(df[c], role, kind)

    n_imputable = sum(1 for c, v in cols.items()
                      if v["role"] == "imputable"
                      and c not in dec["always_observed"])
    block = {
        "complete_path": (str(dec["csv"].relative_to(ROOT))
                          if str(dec["csv"]).startswith(str(ROOT))
                          else str(dec["csv"])),
        "n_rows": int(len(df)),
        "n_columns_total": int(df.shape[1]),
        "n_imputable": n_imputable,
        "identifier_column": ident,
        "downstream_target": target,
        "always_observed": list(dec["always_observed"]),
        "evaluation_coverage": round(
            n_imputable / max(df.shape[1] - 2, 1), 4),
        "provenance_note": dec["provenance"],
        "columns": cols,
    }
    if dec.get("report") and Path(dec["report"]).exists():
        block["build_report"] = str(Path(dec["report"]).relative_to(ROOT))
    return block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    blocks, skipped = {}, []
    for name, dec in DECISIONS.items():
        if not Path(dec["csv"]).exists():
            skipped.append((name, str(dec["csv"])))
            continue
        blocks[name] = build_block(name, dec)

    if skipped:
        # Never silently omit a dataset -- that is the R0 failure mode this
        # project exists to correct.
        for n, p in skipped:
            print(f"[SKIP] {n}: {p} does not exist")
        raise SystemExit(f"{len(skipped)} dataset(s) not built; run the builders first")

    text = CONFIG.read_text()
    head = text.split("\ndatasets:\n")[0]
    body = yaml.safe_dump({"datasets": blocks}, sort_keys=False,
                          allow_unicode=True, width=100, default_flow_style=False)

    print(f"{'dataset':<10} {'rows':>7} {'cols':>5} {'imputable':>10} {'coverage':>9}")
    for n, b in blocks.items():
        print(f"{n:<10} {b['n_rows']:>7} {b['n_columns_total']:>5} "
              f"{b['n_imputable']:>10} {b['evaluation_coverage']*100:>8.1f}%")

    if a.dry_run:
        print("\n[dry-run] datasets.yaml not written")
        return 0
    CONFIG.write_text(head + "\n" + body)
    print(f"\nwrote {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
