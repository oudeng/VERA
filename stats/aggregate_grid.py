"""P2b decision 1: aggregate the grid with the median, and report divergence.

Two rules, and the reason each exists.

**The median is the primary aggregate, for every method and every metric.**
Not because TabCSDI diverges -- because switching the rule for one method is the
kind of inconsistency a reviewer finds first. On NHANES MAR@30 % TabCSDI's five
seeds give R^2 = +0.007, +0.161, -19.35, -224.0, -8.70; a mean over that set is
-50.4 and describes nothing. The median is -8.70, which at least reports a
typical run. `mean +/- sd` is still emitted for the ESM.

**Every method gets a divergence-rate column.** The fraction of runs with
R^2 < 0, i.e. worse than predicting the column mean. This exists because two
separate baselines fail that test and the current presentation hides both:
TabCSDI diverges on three of six datasets (B70), and GAIN is below the line on
all six (B73, median -0.27 to -4.15 across R0's own 300 runs). Reporting a
diverged or clearly outperformed method as a competitor overstates the field, which is the
mirror image of the oracle leakage we removed -- and the review asked for
fairness in both directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent

#: Metrics aggregated across seeds. Higher-is-better flags drive the rank tables.
METRICS = {
    "cont_NRMSE": False, "cont_RMSE": False, "cont_MAE": False,
    "cont_R2": True, "cont_Spearman": True,
    "cat_Accuracy": True, "cat_Macro-F1": True, "cat_Cohen_kappa": True,
}
KEYS = ["dataset", "mechanism", "rate", "method"]


def load_runs(root: Path) -> pd.DataFrame:
    """Every completed cell. Also counts what is *missing*, because a coverage
    gap that nobody prints is exactly finding B3."""
    rows = []
    for f in sorted(root.glob("*/metrics_summary.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception as exc:                      # listed, never dropped
            rows.append({"_unreadable": str(f), "_error": repr(exc)})
    df = pd.DataFrame(rows)
    if "_unreadable" in df.columns:
        bad = df[df._unreadable.notna()]
        if len(bad):
            print(f"WARNING: {len(bad)} unreadable metrics files")
        df = df[df._unreadable.isna()].drop(columns=["_unreadable", "_error"],
                                            errors="ignore")
    return df


def coverage(df: pd.DataFrame, expected_seeds: int = 5) -> pd.DataFrame:
    """Cells with fewer seeds than expected. B3 was a coverage gap the paper
    never disclosed; this makes one impossible to ship unnoticed."""
    g = df.groupby(KEYS).size().rename("n_seeds").reset_index()
    g["complete"] = g.n_seeds >= expected_seeds
    return g



def mask_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Did every seed of a cell see the same mask?

    Recording the mask hash is only useful if something checks it. During P2b a
    mask regeneration landed 7 minutes into a running experiment: same table,
    same row order, different driver set. Nothing raised, and the only reason it
    surfaced was a later run disagreeing with the earlier one. On a 2.3-day grid
    that mistake is easy to repeat and impossible to spot by eye.
    """
    if "mask_md5" not in df.columns:
        return pd.DataFrame()
    g = (df.groupby(["dataset", "mechanism", "rate"], dropna=False)["mask_md5"]
         .agg(n_distinct="nunique", hashes=lambda s: sorted(set(s.dropna())))
         .reset_index())
    g["consistent"] = g.n_distinct <= 1
    return g


def aggregate(df: pd.DataFrame, expected_seeds: int = 5) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(KEYS, dropna=False):
        rec = dict(zip(KEYS, keys))
        rec["n_seeds"] = len(g)
        rec["complete"] = len(g) >= expected_seeds
        # Divergence, for every method alike.
        if "cont_R2" in g:
            r2 = pd.to_numeric(g.cont_R2, errors="coerce")
            rec["divergence_rate"] = float((r2 < 0).mean())
            rec["n_runs_R2_negative"] = int((r2 < 0).sum())
            rec["worst_R2"] = float(r2.min()) if len(r2) else np.nan
        for m in METRICS:
            if m not in g:
                continue
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            if not len(v):
                continue
            rec[f"{m}_median"] = float(v.median())     # primary
            rec[f"{m}_mean"] = float(v.mean())         # ESM
            rec[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rec[f"{m}_min"] = float(v.min())
            rec[f"{m}_max"] = float(v.max())
        if "runtime_sec" in g:
            rec["runtime_median"] = float(pd.to_numeric(g.runtime_sec,
                                                        errors="coerce").median())
        out.append(rec)
    return pd.DataFrame(out).sort_values(KEYS).reset_index(drop=True)


#: `MeanMode` is the reference line this diagnostic is defined against, so its
#: own divergence rate is ~1 by construction: R^2 uses the variance of the true
#: values on the masked cells, while mean imputation fills from the observed
#: cells, and the two means differ slightly. Reading it alongside a *fitted*
#: method that also scores below zero would put a definitional artifact and a
#: real failure in the same column. It is flagged, not dropped.
REFERENCE_METHODS = {"MeanMode"}


def divergence_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset): how often it fails to beat the column mean."""
    rows = []
    for (method, dataset), g in agg.groupby(["method", "dataset"]):
        n = int(g.n_seeds.sum())
        neg = int(g.n_runs_R2_negative.sum()) if "n_runs_R2_negative" in g else 0
        rows.append({
            "method": method, "dataset": dataset, "n_runs": n,
            "n_R2_negative": neg,
            "divergence_rate": neg / n if n else np.nan,
            "worst_R2": float(g.worst_R2.min()) if "worst_R2" in g else np.nan,
            "median_R2": float(g["cont_R2_median"].median())
            if "cont_R2_median" in g else np.nan,
            "is_reference_method": method in REFERENCE_METHODS,
        })
    return (pd.DataFrame(rows)
            .sort_values(["method", "dataset"]).reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(CODE_ROOT / "results" / "P2_main_grid"))
    ap.add_argument("--out", default=str(CODE_ROOT / "results" / "P2_main_grid" / "_agg"))
    ap.add_argument("--expected-seeds", type=int, default=5)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    df = load_runs(root)
    if df.empty:
        print(f"no completed runs under {root}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "long_runs.csv", index=False)

    cov = coverage(df, a.expected_seeds)
    cov.to_csv(out / "coverage.csv", index=False)
    gaps = cov[~cov.complete]

    mc = mask_consistency(df)
    if not mc.empty:
        mc.to_csv(out / "mask_consistency.csv", index=False)
        bad = mc[~mc.consistent]
        if len(bad):
            print("\n!!! MASK INCONSISTENCY -- these cells were run against more "
                  "than one mask, so their seeds are not comparable:")
            print(bad.to_string(index=False))
        else:
            print(f"mask consistency: {len(mc)}/{len(mc)} cells used one mask each")

    agg = aggregate(df, a.expected_seeds)
    agg.to_csv(out / "aggregated_median.csv", index=False)
    div = divergence_table(agg)
    div.to_csv(out / "divergence_rates.csv", index=False)

    print(f"{len(df)} runs over {len(cov)} cells")
    print(f"coverage: {int(cov.complete.sum())}/{len(cov)} cells have "
          f"{a.expected_seeds} seeds")
    if len(gaps):
        print(f"\n{len(gaps)} INCOMPLETE cells (B3 is exactly this, undisclosed):")
        print(gaps.to_string(index=False))
    print("\ndivergence rate by method (R2 < 0 = worse than the column mean):")
    per_method = (div.groupby("method")
                  .apply(lambda g: pd.Series({
                      "n_runs": int(g.n_runs.sum()),
                      "n_negative": int(g.n_R2_negative.sum()),
                      "rate": g.n_R2_negative.sum() / max(g.n_runs.sum(), 1),
                      "worst_R2": float(g.worst_R2.min()),
                      "datasets_affected": int((g.n_R2_negative > 0).sum()),
                      "reference_method": bool(g.is_reference_method.any()),
                  }), include_groups=False)
                  .sort_values("rate", ascending=False))
    print(per_method.to_string())
    if bool(per_method.get("reference_method", pd.Series(dtype=bool)).any()):
        print("\n  NOTE: MeanMode is the reference this metric is defined against;\n"
              "  its rate is ~1 by construction and is not comparable with a\n"
              "  fitted method scoring below the same line.")
    per_method.to_csv(out / "divergence_by_method.csv")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
