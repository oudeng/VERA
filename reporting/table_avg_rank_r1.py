"""R4.1 -- Table 1 (average rank + divergence) for the R1 grid.

The most important table in the paper, so (P4-D) it does not first-run on the
complete grid: `--selftest` exercises the rank machinery on fixtures with
known answers, and any run on an incomplete grid is forced into a scratch
path under a PARTIAL_DO_NOT_USE_ name with a banner caption -- a table built
from 30% of the cells looks entirely plausible, which is exactly what makes
it dangerous. `--final` (the only way to write into reporting/out) refuses
unless every expected run is present and no cell was skipped.

Differences from the R0 exemplar (table_avg_rank.py), each ruled:
  * seed MEDIAN per cell before ranking (A-10), not the mean;
  * a divergence column (share of runs with R^2 < 0), MeanMode flagged as
    the definitional reference (aggregate_grid.REFERENCE_METHODS);
  * a (cell, metric) is ranked only if EVERY expected method has a value --
    partial cells are counted and reported, never silently ranked.

    PYTHONHASHSEED=2025 python reporting/table_avg_rank_r1.py --selftest
    PYTHONHASHSEED=2025 python reporting/table_avg_rank_r1.py \
        --grid results/P2_main_grid --scratch <dir>     # partial dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402
from stats.aggregate_grid import REFERENCE_METHODS, load_runs   # noqa: E402

EXPECTED_METHODS = ["MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE",
                    "HyperImpute", "TabCSDI", "SNI"]
DISPLAY = {"MeanMode": "Mean/Mode"}
METRICS = [("cont_NRMSE", "NRMSE", False), ("cont_MAE", "MAE", False),
           ("cont_R2", "$R^2$", True), ("cat_Accuracy", "Accuracy", True),
           ("cat_Macro-F1", "Macro-F1", True)]
EXPECTED_TOTAL_RUNS = 2565


def build_table(runs: pd.DataFrame, expected=EXPECTED_METHODS):
    """Pure core: (ranks table, diagnostics). Injectable for the selftest."""
    runs = runs.copy()
    runs["cellkey"] = (runs.dataset.astype(str) + "|" + runs.mechanism.astype(str)
                       + "|" + runs.rate.fillna(-1).astype(str))
    records = []            # one row per (cell, metric, method) rank record
    skipped_partial = []    # (cell, metric) with a value-holding method subset
    for (cell, mcol, label, hib) in [
            (c, m, l, h) for c in runs.cellkey.unique()
            for (m, l, h) in METRICS]:
        g = runs[runs.cellkey == cell]
        med = (g.groupby("method")[mcol]
               .apply(lambda v: pd.to_numeric(v, errors="coerce").median())
               .dropna())
        if not len(med):
            continue                       # metric undefined on this cell
        if set(med.index) != set(expected):
            skipped_partial.append((cell, label, sorted(set(expected)
                                                        - set(med.index))))
            continue                       # never rank a partial cell
        ranks = med.rank(ascending=not hib, method="average")
        for meth, r in ranks.items():
            records.append({"cell": cell, "metric": label, "method": meth,
                            "rank": float(r)})
    rec = pd.DataFrame(records)
    diag = {"n_rank_records": len(rec),
            "n_cells_skipped_partial": len(skipped_partial),
            "skipped_examples": skipped_partial[:5]}
    if not len(rec):
        return pd.DataFrame(), diag

    table = (rec.groupby(["method", "metric"])["rank"].mean()
             .unstack("metric").reindex(expected)
             [[l for (_, l, _) in METRICS if l in rec.metric.unique()]])
    table["Overall"] = rec.groupby("method")["rank"].mean().reindex(expected)

    r2 = pd.to_numeric(runs.cont_R2, errors="coerce")
    div = (runs.assign(_neg=r2 < 0, _has=r2.notna())
           .groupby("method")[["_neg", "_has"]].sum())
    table["Neg.-$R^2$"] = (div._neg / div._has).reindex(expected)
    diag["settings_per_metric"] = (rec.groupby("metric")["cell"].nunique()
                                   .to_dict())
    return table, diag


def emit(table: pd.DataFrame, diag: dict, out_path: Path, *,
         partial_banner: str | None, grid_path: str) -> Path:
    body = table.reset_index().rename(columns={"index": "method"})
    body["method"] = body["method"].map(
        lambda m: DISPLAY.get(m, m) + (r"$^{\dagger}$"
                                       if m in REFERENCE_METHODS else ""))
    for c in body.columns[1:]:
        if c == "Neg.-$R^2$":
            body[c] = body[c].map(lambda v: rf"{100*v:.0f}\%"
                                  if pd.notna(v) else "---")
        else:
            body[c] = body[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "---")
    caption = (r"Average rank across settings (seed medians; lower is "
               r"better) with per-method divergence rate.")
    if partial_banner:
        caption = (r"\textbf{" + partial_banner + r"} " + caption)
    style = TableStyle(environment="table*", notes=(
        r"\textit{Protocol:} within each dataset $\times$ mechanism $\times$ "
        r"rate setting, the five seeds are reduced to their median before "
        r"ranking (every method and metric alike); a setting enters a "
        r"metric's ranking only when all nine methods report it. The Overall "
        r"column pools every (setting, metric) rank record. Neg.-$R^2$ is the "
        r"share of a method's runs with $R^2<0$; $^{\dagger}$ marks the "
        r"reference method for which a near-1 rate is definitional rather "
        r"than a failure.",))
    up, down = r"$\uparrow$", r"$\downarrow$"
    metric_heads = [f"{l} {up if h else down}" for (_, l, h) in METRICS
                    if l in table.columns]
    tex = dataframe_to_tex(
        body, caption=caption, label="tab:avg_rank_r1",
        header=["Method"] + metric_heads + ["Overall", "Neg.-$R^2$"],
        style=style, escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_avg_rank_r1.py",
        "input": grid_path, "code_SNI commit": runconfig.git_commit(),
        **{k: str(v) for k, v in diag.items()}})


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    meths = ["A", "B", "C", "D"]
    rows = []
    for ds in ("X", "Y"):
        for s in (1, 2):
            for m in meths:
                rows.append({"dataset": ds, "mechanism": "MAR", "rate": 0.3,
                             "seed": s, "method": m,
                             "cont_NRMSE": 0.5, "cont_MAE": 0.4,
                             "cont_R2": 0.2, "cat_Accuracy": np.nan,
                             "cat_Macro-F1": np.nan})
    t, d = build_table(pd.DataFrame(rows), expected=meths)
    check(np.allclose(t["NRMSE"], 2.5), "all-tied ranks = (m+1)/2 = 2.5")
    check(d["n_cells_skipped_partial"] == 0 and d["n_rank_records"] == 24,
          "tie case: 2 cells x 3 defined metrics x 4 methods = 24 records")
    check("Accuracy" not in t.columns, "all-NaN metric dropped, not ranked")

    rows2 = [dict(r) for r in rows]
    for r in rows2:                       # A dominates; direction-aware
        if r["method"] == "A":
            r.update(cont_NRMSE=0.1, cont_MAE=0.1, cont_R2=0.9)
    t2, _ = build_table(pd.DataFrame(rows2), expected=meths)
    check(t2.loc["A", "Overall"] == 1.0, "dominant method Overall rank 1.0")
    check(t2.loc["A", "$R^2$"] == 1.0,
          "higher-is-better direction respected for R2")

    rows3 = [r for r in rows2 if not (r["dataset"] == "Y"
                                      and r["method"] == "D")]
    t3, d3 = build_table(pd.DataFrame(rows3), expected=meths)
    check(d3["n_cells_skipped_partial"] == 3
          and d3["n_rank_records"] == 12,
          "missing-method cell skipped per metric (3 skips), never ranked")
    check(t3.loc["A", "Overall"] == 1.0, "ranks from complete cells only")

    rows4 = [dict(r) for r in rows]
    for i, r in enumerate(rows4):
        if r["method"] == "B":
            r["cont_R2"] = -0.5 if r["seed"] == 1 else 0.5
    t4, _ = build_table(pd.DataFrame(rows4), expected=meths)
    check(abs(t4.loc["B", "Neg.-$R^2$"] - 0.5) < 1e-12,
          "divergence rate = share of runs with R2<0 (0.5)")
    check(abs(t4.loc["A", "Neg.-$R^2$"] - 0.0) < 1e-12, "clean method Neg.-R2 0")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(CODE_ROOT / "results" / "P2_main_grid"))
    ap.add_argument("--scratch", default=None,
                    help="REQUIRED for partial grids; output goes here with a "
                         "PARTIAL_DO_NOT_USE_ prefix")
    ap.add_argument("--final", action="store_true",
                    help="write reporting/out/tab_avg_rank_r1.tex; refuses "
                         "unless the grid is complete and nothing was skipped")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    runs = load_runs(Path(a.grid))
    n = len(runs)
    complete = n >= EXPECTED_TOTAL_RUNS
    table, diag = build_table(runs)
    diag["n_runs_loaded"] = n

    if a.final:
        if not complete or diag["n_cells_skipped_partial"]:
            print(f"REFUSING --final: runs {n}/{EXPECTED_TOTAL_RUNS}, "
                  f"skipped cells {diag['n_cells_skipped_partial']}. A "
                  f"partial Table 1 looks plausible; that is why this flag "
                  f"exists.", file=sys.stderr)
            return 2
        out = emit(table, diag, CODE_ROOT / "reporting" / "out"
                   / "tab_avg_rank_r1.tex", partial_banner=None,
                   grid_path=a.grid)
    else:
        if not a.scratch:
            print("REFUSING: partial run without --scratch. Never write a "
                  "partial Table 1 anywhere it could be mistaken for real.",
                  file=sys.stderr)
            return 2
        pct = 100 * n / EXPECTED_TOTAL_RUNS
        out = emit(table, diag, Path(a.scratch)
                   / "PARTIAL_DO_NOT_USE_tab_avg_rank_r1.tex",
                   partial_banner=(f"PARTIAL GRID ({n}/{EXPECTED_TOTAL_RUNS} "
                                   f"runs, {pct:.0f}\\%) -- NOT FOR "
                                   f"PUBLICATION."),
                   grid_path=a.grid)
    print(f"[OK] wrote {out}")
    print({k: diag[k] for k in ("n_runs_loaded", "n_rank_records",
                                "n_cells_skipped_partial")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
