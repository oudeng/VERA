"""P2b decision 2(c2): measure NHANES's real MAR mechanism on the UNFILTERED table.

This analysis takes no part in any imputation evaluation. It exists so that three
claims in the paper rest on measurement rather than on argument:

1. **NHANES has a documented, verifiable MAR mechanism.** The survey runs the
   fasting-dependent assays only on the morning fasting subsample, so fasting
   status -- recorded for every participant -- determines whether those assays
   exist. That is missingness at random by construction, and it is the only
   mechanism in the benchmark that can be checked against a source.

2. **Our simulated coefficients are calibrated, not invented.** The effect size
   measured here is what the `age` and `smoking_std` slopes in
   `configs/missingness.yaml` are scaled against, and this script emits the
   table that goes into the ESM specification.

3. **The complete-case construction destroys the mechanism** (finding B72). The
   rows that exhibit it are exactly the rows a complete-case filter discards:
   46.8 % of adults in the source cycle are fasting, 95.9 % of the benchmark
   table are. That observation generalises to every "complete-case then re-mask"
   benchmark in the imputation literature, which is why it is prepared as a
   contribution rather than a limitation.

The measurement is made on the adult merge BEFORE `dropna`, which is the only
place the mechanism is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "T2b_nhanes_mechanism"

#: Assays the survey protocol restricts to the fasting subsample, and the
#: comparison columns that have no fasting requirement.
FASTING_DEPENDENT = ["fasting_glucose", "triglycerides"]
FASTING_INDEPENDENT = ["hba1c", "hdl_cholesterol", "systolic_bp", "bmi",
                       "waist_circumference"]


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def measure(adults: pd.DataFrame) -> dict:
    """Per-column missingness split by fasting status, on unfiltered adults."""
    fast = adults.fasting_state_std
    known = fast.notna()
    a, b = adults[known & (fast == 1)], adults[known & (fast == 0)]

    rows = []
    for col in FASTING_DEPENDENT + FASTING_INDEPENDENT:
        if col not in adults.columns:
            continue
        pf, pn = float(a[col].isna().mean()), float(b[col].isna().mean())
        rows.append({
            "column": col,
            "fasting_dependent": col in FASTING_DEPENDENT,
            "missing_if_fasting": pf,
            "missing_if_not_fasting": pn,
            "gap_pp": 100.0 * (pn - pf),
            "logodds_contrast": _logit(pn) - _logit(pf),
        })
    df = pd.DataFrame(rows)

    dep = df[df.fasting_dependent]
    indep = df[~df.fasting_dependent]
    return {
        "n_adults_unfiltered": int(len(adults)),
        "n_fasting_known": int(known.sum()),
        "prevalence_fasting_unfiltered": float((fast == 1).sum() / max(known.sum(), 1)),
        "table": df,
        "effect_size": {
            "fasting_dependent_mean_contrast": float(dep.logodds_contrast.mean()),
            "fasting_independent_mean_contrast": float(indep.logodds_contrast.mean())
            if len(indep) else float("nan"),
            "ratio": (float(dep.logodds_contrast.mean()
                            / indep.logodds_contrast.mean())
                      if len(indep) and indep.logodds_contrast.mean() != 0
                      else float("nan")),
        },
    }


def quantify_destruction(adults: pd.DataFrame, complete: pd.DataFrame) -> dict:
    """How much of the mechanism survives the complete-case step (B72)."""
    fast_before = adults.fasting_state_std
    known = fast_before.notna()
    p_before = float((fast_before == 1).sum() / max(known.sum(), 1))
    p_after = float((complete.fasting_state_std == 1).mean())

    def _sd_of_binary(p):        # the driver's usable variation
        return float(np.sqrt(p * (1 - p)))

    return {
        "prevalence_before": p_before,
        "prevalence_after": p_after,
        "n_before": int(known.sum()), "n_after": int(len(complete)),
        "non_fasting_before": int((fast_before == 0).sum()),
        "non_fasting_after": int((complete.fasting_state_std == 0).sum()),
        "driver_sd_before": _sd_of_binary(p_before),
        "driver_sd_after": _sd_of_binary(p_after),
        "sd_retained_fraction": (_sd_of_binary(p_after) / _sd_of_binary(p_before)
                                 if _sd_of_binary(p_before) > 0 else float("nan")),
        "decile_split_degenerate_after": bool(p_after > 0.9 or p_after < 0.1),
        "note": "A driver whose top and bottom deciles are the same rows cannot "
                "produce a contrast at all; the achieved value is undefined, not "
                "small. See missingness/diagnostics.logodds_contrast.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "nhanes"))
    ap.add_argument("--complete", default=str(ROOT / "data/derived/NHANES_complete.csv"))
    a = ap.parse_args()

    # Rebuild the adult merge exactly as the builder does, but stop before the
    # complete-case step -- that is the only place the mechanism is visible.
    import sys
    sys.path.insert(0, str(ROOT))
    from data_layer.build_nhanes import load_modules, derive

    report: dict = {"raw_dir": a.raw_dir, "modules": {}}
    merged = load_modules(Path(a.raw_dir), report)
    adults = derive(merged, report)

    complete = pd.read_csv(a.complete)
    m = measure(adults)
    d = quantify_destruction(adults, complete)

    OUT.mkdir(parents=True, exist_ok=True)
    m["table"].to_csv(OUT / "mechanism_by_column.csv", index=False)
    payload = {"measurement": {k: v for k, v in m.items() if k != "table"},
               "table": m["table"].to_dict("records"),
               "complete_case_destruction": d}
    (OUT / "nhanes_mechanism.json").write_text(json.dumps(payload, indent=2))

    print(f"unfiltered adults: {m['n_adults_unfiltered']}, "
          f"fasting status known for {m['n_fasting_known']}, "
          f"prevalence fasting {m['prevalence_fasting_unfiltered']:.3f}")
    print()
    print(m["table"].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print()
    e = m["effect_size"]
    print(f"mean log-odds contrast: fasting-dependent assays "
          f"{e['fasting_dependent_mean_contrast']:+.3f}, "
          f"fasting-independent {e['fasting_independent_mean_contrast']:+.3f} "
          f"(ratio {e['ratio']:.1f}x)")
    print()
    print("complete-case destruction (B72):")
    print(f"  fasting prevalence {d['prevalence_before']:.3f} -> {d['prevalence_after']:.3f}")
    print(f"  non-fasting participants {d['non_fasting_before']} -> {d['non_fasting_after']}")
    print(f"  driver sd retained: {100*d['sd_retained_fraction']:.1f}%")
    print(f"  decile split degenerate after filtering: {d['decile_split_degenerate_after']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
