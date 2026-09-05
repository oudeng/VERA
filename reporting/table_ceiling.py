"""T4H.1 -- reliability-ceiling table for the ESM (self-attack analysis).

Input: results/T4_ceiling/ceiling_analysis.json. Guard: every attainment MUST
carry its bootstrap CI -- the ceiling is two divisions and a square root away
from noisy inputs, and a bare point estimate here would be exactly the
overclaim this analysis exists to prevent (P4-H: 'no naked point estimates').

    PYTHONHASHSEED=2025 python reporting/table_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

CEIL = CODE_ROOT / "results" / "T4_ceiling"
DATASETS = ["MIMIC", "eICU"]
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest"]
DISPLAY = {"P": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           "SNI-D": r"SNI \Dm{}"}


def build(out_path: Path) -> Path:
    dd = json.loads((CEIL / "ceiling_analysis.json").read_text())
    d = {ds: dd[ds]["row"] for ds in DATASETS}       # PRIMARY: Table-5 caliber
    dm = {ds: dd[ds]["matrix"] for ds in DATASETS}   # secondary: robustness
    for ds in DATASETS:                       # the no-naked-point guard
        for blk in (d[ds], dm[ds]):
            if "ceiling_ci95" not in blk:
                raise ValueError(f"{ds}: ceiling lacks a CI; refusing a bare "
                                 f"point estimate (P4-H).")
            for o in OBJECTS:
                if "attainment_ci95" not in blk[o]:
                    raise ValueError(f"{ds}/{o}: attainment lacks a CI.")

    rows = []
    for o in OBJECTS:
        row = {"obj": DISPLAY.get(o, o)}
        for ds in DATASETS:
            x = d[ds][o]
            lo, hi = x["attainment_ci95"]
            row[f"{ds}_rho"] = f"{x['rho_XA']:.3f}"
            row[f"{ds}_att"] = (rf"{100*x['attainment']:.0f}\% "
                                rf"[{100*lo:.0f}, {100*hi:.0f}]")
        rows.append(row)
    rows.append({"obj": "Permutation-on-SNI",
                 **{f"{ds}_rho": r"$\equiv 1$" for ds in DATASETS},
                 **{f"{ds}_att": "(circular; excluded)" for ds in DATASETS}})
    body = pd.DataFrame(rows)

    hdrbits = []
    for ds in DATASETS:
        b = d[ds]
        clo, chi = b["ceiling_ci95"]
        hdrbits.append(
            rf"{ds}: $r_{{xx}}={b['r_xx']:.2f}$, "
            rf"$\rho_{{AA}}={b['rho_AA']:.2f}$, ceiling "
            rf"$=\sqrt{{\rho_{{AA}}/r_{{xx}}}}={b['ceiling']:.2f}$ "
            rf"[{clo:.2f}, {chi:.2f}]")
    mm = "; ".join(
        rf"{ds}: ceiling {dm[ds]['ceiling']:.2f}, SNI \Dm{{}} "
        rf"{100*dm[ds]['SNI-D']['attainment']:.0f}\% "
        rf"[{100*dm[ds]['SNI-D']['attainment_ci95'][0]:.0f}, "
        rf"{100*dm[ds]['SNI-D']['attainment_ci95'][1]:.0f}]"
        for ds in DATASETS)
    style = TableStyle(environment="table", notes=(
        r"\textit{Setup:} " + "; ".join(hdrbits) + r". "
        r"Attainment $=\rho(X,A)/\sqrt{\rho_{AA}}$, the fraction of the "
        r"seed-invariant-artifact ceiling an object reaches; bootstrap 95\% "
        r"CIs resample the five seeds. \textit{Caliber (the reliability-ceiling protocol):} every "
        r"quantity here is ROW-level -- per-target row correlations pooled, "
        r"then the median -- the exact unit and aggregation of the main "
        r"faithfulness table, whose $\rho(X,A)$ values are therefore "
        r"reproduced in this table's first columns and can be divided by "
        r"$\sqrt{\rho_{AA}}$ by hand to re-derive the attainments printed. "
        r"The matrix-level caliber (flattened target rows) gives uniformly "
        r"LOWER attainments and is kept as a robustness check: " + mm + r". "
        r"\textit{Approximation stated:} classical disattenuation assumes "
        r"additive independent error in a Pearson framework and is applied "
        r"here to Spearman correlations. The reading is unchanged in both "
        r"calibers: the host's limited reproducibility lowers the ceiling, "
        r"and the attention matrix still reaches only a third to a half of "
        r"it -- a gap the unstable host cannot explain. No post-hoc readout "
        r"is uniformly best across datasets in either caliber. "
        r"\textit{Provenance:} this ceiling analysis was specified after "
        r"the primary results were known. It interprets the prospectively specified "
        r"verdict rather than testing it, and no verdict in this paper "
        r"depends on it.",))
    tex = dataframe_to_tex(
        body,
        caption=(r"Reliability ceiling for faithfulness, and each object's "
                 r"attainment of it."),
        label="tab:ceiling",
        column_format="lcccc",
        header=["Audit object", r"MIMIC $\rho(X,A)$", r"attain.\ [CI]",
                r"eICU $\rho(X,A)$", r"attain.\ [CI]"],
        style=style, escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_ceiling.py",
        "input": str(CEIL / "ceiling_analysis.json"),
        "code_SNI commit": runconfig.git_commit()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_ceiling.tex"))
    a = ap.parse_args()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
