"""Generate Fig. 3 (multi-rate sensitivity) as a PDF the manuscript includes (W3).

Published layout (`fig:multirate`): 2 metric rows (NRMSE, Macro-F1) x 3
mechanism columns (MCAR, MAR, MNAR); x = missing rate 10-50 %; line = the
cross-dataset average; band = +-1 std across datasets; SNI, MissForest and
MIWAE highlighted, the rest de-emphasized.

R1 differences enforced here rather than remembered:

  * every method appears in every panel it has data for -- R0's MCAR/MAR
    panels had only the 30 % point for the non-SNI methods and the caption did
    not say so (finding B3), and the MNAR panels dropped HyperImpute/TabCSDI
    (R2-6b). The generator REFUSES to draw a curve from a single rate point
    unless the caller passes --allow-partial, in which case partial curves are
    drawn dashed and the omission is printed loudly;
  * the per-(dataset, cell) value is the seed MEDIAN (P2b decision 1 / A-10),
    not the mean;
  * CDC2022 runs a slimmed grid (MCAR/MAR at 30 % only, P2c section 4) and is
    excluded from cross-rate averaging by default, stated in the caption note.

Input: `long_runs.csv` from stats/aggregate_grid.py (one row per completed
run). [[GRID: regenerate when the grid completes]]

    python reporting/fig_multirate.py --long results/P2_main_grid_agg/long_runs.csv \
        --out reporting/out/Fig_multirate_sensitivity.pdf
    python reporting/fig_multirate.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.3, 0.5]
METRICS = [("cont_NRMSE", r"NRMSE $\downarrow$"),
           ("cat_Macro-F1", r"Macro-F1 $\uparrow$")]
HIGHLIGHT = {"SNI": "tab:red", "MissForest": "tab:blue", "MIWAE": "tab:green"}
EXCLUDE_DATASETS = ["CDC2022"]          # slim grid: no cross-rate curve exists


def build(long_path: Path, out_path: Path, allow_partial: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(long_path)
    df = df[~df.dataset.isin(EXCLUDE_DATASETS) & df.rate.notna()
            & (df.mechanism != "REAL_PATTERN")]
    methods = sorted(df.method.unique())

    fig, axes = plt.subplots(len(METRICS), len(MECHANISMS),
                             figsize=(11, 6.2), sharex=True)
    partial_notes = []
    for r, (metric, mlabel) in enumerate(METRICS):
        for c, mech in enumerate(MECHANISMS):
            ax = axes[r, c]
            g = df[df.mechanism == mech].dropna(subset=[metric])
            for method in methods:
                gm = g[g.method == method]
                # seed median per (dataset, rate) -- A-10 -- then stats across datasets
                cell = gm.groupby(["dataset", "rate"])[metric].median().reset_index()
                stat = cell.groupby("rate")[metric].agg(["mean", "std", "count"])
                stat = stat.reindex(RATES).dropna(subset=["mean"])
                if len(stat) == 0:
                    continue
                if len(stat) < len(RATES):
                    msg = (f"{method}/{mech}/{metric}: data at "
                           f"{list(stat.index)} of {RATES}")
                    if not allow_partial:
                        raise ValueError(
                            f"partial curve -- {msg}. R0 published exactly this "
                            f"as a full trend (B3); pass --allow-partial to draw "
                            f"it dashed instead.")
                    partial_notes.append(msg)
                hi = method in HIGHLIGHT
                ax.plot(stat.index, stat["mean"],
                        marker="o", ms=3.5 if hi else 2.5,
                        lw=1.8 if hi else 0.9,
                        ls="-" if len(stat) == len(RATES) else "--",
                        color=HIGHLIGHT.get(method, "0.72"),
                        zorder=3 if hi else 2,
                        label=method if (r == 0 and c == 0) else None)
                sd = stat["std"].fillna(0.0)
                ax.fill_between(stat.index, stat["mean"] - sd, stat["mean"] + sd,
                                color=HIGHLIGHT.get(method, "0.85"),
                                alpha=0.15 if hi else 0.06, zorder=1)
            if r == 0:
                ax.set_title(mech)
            if c == 0:
                ax.set_ylabel(mlabel)
            if r == len(METRICS) - 1:
                ax.set_xlabel("missing rate")
            ax.set_xticks(RATES)
            ax.grid(alpha=0.25, lw=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = np.argsort([m not in HIGHLIGHT for m in labels], kind="stable")
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=min(9, len(labels)), loc="lower center", frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)
    for msg in partial_notes:
        print(f"[partial] {msg}")
    print(f"[OK] wrote {out_path} "
          f"({len(methods)} methods, datasets excl. {EXCLUDE_DATASETS})")
    return out_path


def _selftest() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for m in ["SNI", "MissForest", "MIWAE", "MeanMode", "KNN", "MICE",
              "GAIN", "HyperImpute", "TabCSDI"]:
        for ds in ["A", "B", "C", "D", "E", "F"]:
            for mech in MECHANISMS:
                for rate in RATES:
                    for seed in (1, 2, 3, 5, 8):
                        rows.append({"method": m, "dataset": ds,
                                     "mechanism": mech, "rate": rate,
                                     "seed": seed,
                                     "cont_NRMSE": rng.uniform(.1, .5) + rate * .3,
                                     "cat_Macro-F1": rng.uniform(.4, .9) - rate * .2})
    tmp = Path("/tmp") / "fig3_selftest_long.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    out = build(tmp, CODE_ROOT / "reporting" / "out"
                / "Fig_multirate_SELFTEST.pdf", allow_partial=False)
    assert out.exists() and out.stat().st_size > 10_000
    print("[OK] selftest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long")
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "Fig_multirate_sensitivity.pdf"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.long:
        ap.error("--long required (or --selftest)")
    build(Path(a.long), Path(a.out), a.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
