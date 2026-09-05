"""Generate Table 3 (cross-seed stability of D) as an `\\input`-able .tex (W3).

Reproduces the published layout of `tab:d_stability_extended` (R0 main text):

    Dataset | d | N_pairs | Mean rho | Min rho | Max rho | % pairs p<0.05

with this revision's numbers. Input is one or more T2f-format pairs files
(`results/T2f_d_stability/pairs_seed_{ds}.csv`; columns a,b,spearman,p,...)
plus the corresponding D matrix for the feature count d. Rows are data-driven,
so re-running after the grid (or adding datasets for the R2-4 width story)
changes the table without touching this file.

R1 protocol differences that the caption must carry (comparison_registry):
the eICU/MIMIC tables are rebuilt (column sets differ from R0), so the R0
published 0.951 / 0.633 are NOT directly comparable and are deliberately not
printed side by side here.

    PYTHONHASHSEED=2025 python reporting/table_d_stability.py \
        --out reporting/out/tab_d_stability.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

STAB = CODE_ROOT / "results" / "T2f_d_stability"


def build(datasets: list[str], out_path: Path, pairs_dir: Path) -> Path:
    rows = []
    for ds in datasets:
        pairs = pd.read_csv(pairs_dir / f"pairs_seed_{ds}.csv")
        D = pd.read_csv(pairs_dir / f"D_{ds}_seed1_cpu_t2.csv", index_col=0)
        rows.append({
            "Dataset": ds,
            "d": len(D),
            "N": len(pairs),
            "mean": f"{pairs.spearman.mean():.3f}",
            "min": f"{pairs.spearman.min():.3f}",
            "max": f"{pairs.spearman.max():.3f}",
            "sig": rf"{100 * (pairs.p < 0.05).mean():.0f}\%",
        })
    body = pd.DataFrame(rows)

    style = TableStyle(
        environment="table*",
        size=r"\scriptsize",
        notes=(
            r"\textit{Protocol and interpretation:} values are pairwise Spearman "
            r"$\rho$ for flattened off-diagonal entries across five random seeds "
            r"($\binom{5}{2}=10$ pairs) under MAR 30\%, with the this revision protocol "
            r"(200 epochs, early stopping disabled, CPU, BLAS threads pinned at 2). "
            r"The this revision tables are rebuilt relative to the original submission (target and zero-variance "
            r"columns removed), so these values are not directly comparable to the "
            r"figures published in the original submission; the correspondence is registered in the "
            r"comparison registry. [[GRID: extend/recompute from the full grid if "
            r"the final protocol draws these from grid cells]]",
        ),
    )
    tex = dataframe_to_tex(
        body,
        caption=r"Cross-seed stability of the reliance matrix $\mathbf{D}$.",
        label="tab:d_stability_extended",
        header=["Dataset", "$d$", r"$N_{\text{pairs}}$", r"Mean $\rho$",
                r"Min $\rho$", r"Max $\rho$", r"\% pairs $p<0.05$"],
        style=style,
        escape_data=False,
    )
    return write_tex(out_path, tex, provenance={
        "generator": "code_SNI/reporting/table_d_stability.py",
        "input": str(pairs_dir),
        "code_SNI commit": runconfig.git_commit(),
        "datasets": ",".join(datasets),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["MIMIC", "eICU"])
    ap.add_argument("--pairs-dir", default=str(STAB))
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_d_stability.tex"))
    a = ap.parse_args()
    out = build(a.datasets, Path(a.out), Path(a.pairs_dir))
    print(f"[OK] wrote {out}")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
