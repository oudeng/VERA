"""T2b.1: is the MIMIC/eICU schema convergence forced, self-inflicted, or an asset?

The P2 checkpoint reported that both ICU tables land on "the same 16 columns".
That was imprecise -- after normalizing the naming conventions the two share 15
features, and each keeps one the other lacks (`charlson_index` against `gcs`).
The overlap is still high enough that a reviewer will notice, so it has to be
characterized rather than mentioned.

Three questions, in the order the P2b instruction poses them:

1. Did we cause the convergence? Every deletion is traced to the finding that
   mandated it, and the source tables are compared before any cleaning. If the
   overlap predates us it is a property of the extraction templates; if we
   deleted a clean column unique to one table, we over-pruned and must restore it.
2. Same schema is not same data. Standardized mean differences and KS distances
   per shared column, plus cohort-level differences.
3. What is that worth for R2-1? A cross-database dependency-recovery test is
   only meaningful if the columns mean the same thing while the populations
   differ, which is exactly what (1) and (2) establish.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "T2b_schema_review"

#: Naming conventions differ between the two extraction scripts; strip the
#: cosmetic suffixes so `MAP_mmHg` and `map_mmhg` are recognized as one variable.
_SUFFIXES = ("_std", "_mmhg", "_mmol_l", "_mg_dl", "_years", "_min", "_max")


def canon(col: str) -> str:
    c = col.lower()
    for s in _SUFFIXES:
        if c.endswith(s):
            c = c[: -len(s)]
    return c


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, no SciPy dependency."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference, pooled sd -- the usual cohort-comparison
    effect size. |SMD| > 0.1 is the conventional threshold for imbalance."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def review(mimic: pd.DataFrame, eicu: pd.DataFrame) -> dict:
    mc = {canon(c): c for c in mimic.columns}
    ec = {canon(c): c for c in eicu.columns}
    shared = sorted(set(mc) & set(ec) - {"id"})

    rows = []
    for k in shared:
        a = pd.to_numeric(mimic[mc[k]], errors="coerce").to_numpy(float)
        b = pd.to_numeric(eicu[ec[k]], errors="coerce").to_numpy(float)
        rows.append({
            "variable": k,
            "mimic_column": mc[k], "eicu_column": ec[k],
            "mimic_mean": float(np.nanmean(a)), "eicu_mean": float(np.nanmean(b)),
            "mimic_sd": float(np.nanstd(a, ddof=1)),
            "eicu_sd": float(np.nanstd(b, ddof=1)),
            "smd": smd(a, b), "ks": ks_distance(a, b),
        })
    df = pd.DataFrame(rows).sort_values("ks", ascending=False)

    return {
        "shared_variables": shared,
        "mimic_only": sorted(set(mc) - set(ec)),
        "eicu_only": sorted(set(ec) - set(mc)),
        "n_shared": len(shared),
        "table": df,
        "cohort": {
            "mimic_n": int(len(mimic)), "eicu_n": int(len(eicu)),
            "n_ratio": round(len(mimic) / len(eicu), 3),
        },
        "distribution_summary": {
            "n_variables": len(df),
            "n_smd_above_0.1": int((df.smd.abs() > 0.1).sum()),
            "n_smd_above_0.5": int((df.smd.abs() > 0.5).sum()),
            "n_ks_above_0.1": int((df.ks > 0.1).sum()),
            "n_ks_above_0.3": int((df.ks > 0.3).sum()),
            "median_ks": float(df.ks.median()),
            "max_ks": float(df.ks.max()),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", default=str(ROOT / "data/derived/MIMIC_complete.csv"))
    ap.add_argument("--eicu", default=str(ROOT / "data/derived/eICU_complete.csv"))
    a = ap.parse_args()

    m, e = pd.read_csv(a.mimic), pd.read_csv(a.eicu)
    rep = review(m, e)
    OUT.mkdir(parents=True, exist_ok=True)
    rep["table"].to_csv(OUT / "shared_column_distributions.csv", index=False)

    print(f"MIMIC {m.shape} vs eICU {e.shape}")
    print(f"shared variables (canonical names, ID excluded): {rep['n_shared']}")
    print(f"  MIMIC only : {rep['mimic_only']}")
    print(f"  eICU  only : {rep['eicu_only']}")
    print()
    t = rep["table"]
    print(t[["variable", "mimic_mean", "eicu_mean", "smd", "ks"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    print()
    s = rep["distribution_summary"]
    print(f"|SMD| > 0.1 : {s['n_smd_above_0.1']}/{s['n_variables']}   "
          f"|SMD| > 0.5 : {s['n_smd_above_0.5']}/{s['n_variables']}")
    print(f"KS    > 0.1 : {s['n_ks_above_0.1']}/{s['n_variables']}   "
          f"KS    > 0.3 : {s['n_ks_above_0.3']}/{s['n_variables']}   "
          f"median KS {s['median_ks']:.3f}, max {s['max_ks']:.3f}")

    payload = {k: v for k, v in rep.items() if k != "table"}
    payload["table"] = rep["table"].to_dict("records")
    (OUT / "schema_review.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
