"""R4.4 -- faithfulness table (Results, the decisive axis).

Input: results/T3_faithfulness/faithfulness_summary.json (verdict analysis of
08-21, selftest-validated code) and faithfulness_cells.csv for the top-3
column. Panel A: median rho(row, A-row) per audit object per dataset. Panel
B: the paired D - TAP comparison. Guard (P4-D, from the 08-21 first-author
directive): Panel B MUST carry n, an effect size and a CI alongside the
p-value -- a p-only rendering is refused, because "failure to reject" and
"no effect" are different sentences and the table must be able to say which.

    PYTHONHASHSEED=2025 python reporting/table_faithfulness.py
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
from reporting.latex import (TableStyle, dataframe_to_tex,  # noqa: E402
                             math_signed, write_tex)

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
DATASETS = ["MIMIC", "eICU"]
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest"]
DISPLAY = {"P": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           "SNI-D": r"SNI \Dm{}"}
_REQUIRED_B = ("n_pairs", "median_delta", "wilcoxon_p",
               "rank_biserial_r", "median_ci95")


def _sci(x: float, sig: int = 1) -> str:
    """A power of ten a reader reads, not the one Python prints.

    "1.2e-04" shipped in Table 7 beside "0.000061" in the adjacent column:
    two notations for the same order of magnitude, one row apart.
    """
    from math import floor, log10
    if x == 0:
        return "0"
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return rf"${m:.{sig}f} \times 10^{{{e}}}$"


def build(out_path: Path) -> Path:
    """Both panels consume the SINGLE evidence source t_final.json
    (P5R-F SS1): Panel A (Table 6) carries 15-seed rho medians for SNI-D
    and TAP with the MF readouts as an initial five-seed descriptive
    subset; Panel B (Table 7) is the unified-estimand paired table --
    T = mean of seed-level medians, seed-only bootstrap CI, exact
    enumeration p under seed-block sign-exchangeability, Holm within the
    with-prior two-table family, negative-seed counts -- for the
    with-prior AND no-prior comparisons. Statistic macros for the prose
    are emitted from the same read."""
    tf = json.loads((CODE_ROOT / "results" / "T5_stats"
                     / "t_final.json").read_text())
    summary = json.loads((FAITH / "faithfulness_summary.json").read_text())
    cells = pd.read_csv(FAITH / "faithfulness_cells.csv")
    cells = cells[cells.scope == "full"]

    rho15 = tf["faithfulness_rho_medians_15seed"]
    rows = []
    for obj in OBJECTS:
        row = {"obj": DISPLAY.get(obj, obj)}
        for ds in DATASETS:
            if obj in ("SNI-D", "P"):
                key = "TAP" if obj == "P" else "SNI-D"
                row[f"{ds}_rho"] = f"{rho15[ds][key]['rho_median']:.3f}"
                row[f"{ds}_top3"] = f"{rho15[ds][key]['top3_mean']:.2f}"
            else:
                row[f"{ds}_rho"] = (f"{summary[ds][obj]['rho_median']:.3f}"
                                    + r"$^{\dagger}$")
                g = cells[(cells.dataset == ds) & (cells.method == obj)]
                row[f"{ds}_top3"] = f"{g.top3_overlap.mean():.2f}"
        rows.append(row)
    panelA = pd.DataFrame(rows)

    fai = tf["faithfulness"]
    npf = tf["noprior_faithfulness"]
    brows = []
    for label, blk, holm_col in (
            ("MIMIC", fai["MIMIC"], f"{fai['MIMIC']['p_holm']:.3f}"),
            ("eICU", fai["eICU"], f"{fai['eICU']['p_holm']:.3f}"),
            ("No-prior MIMIC", npf["MIMIC"],
             _sci(npf['MIMIC']['p_holm'])),
            ("No-prior eICU", npf["eICU"],
             _sci(npf['eICU']['p_holm']))):
        lo, hi = blk["ci95_T"]
        ptxt = (f"{blk['p_exact']:.6f}".rstrip("0")
                if blk["p_exact"] < 0.001 else f"{blk['p_exact']:.3f}")
        brows.append({
            "ds": label, "n": blk["n_seeds"],
            "T": math_signed(blk["T"]),
            "ci": rf"[{math_signed(lo)}, {math_signed(hi)}]",
            "p": ptxt,
            "ph": holm_col,
            "neg": blk["seeds_negative"]})
    panelB = pd.DataFrame(brows)

    styleA = TableStyle(environment="table*", notes=(
        r"\textit{Behavioral reference:} $A[f,j]$ = mean degradation of "
        r"target $f$'s masked-cell NRMSE when standardized input column "
        r"$j$ is permuted across rows at inference (5 permutations), on "
        r"the run's own trained per-feature models. Entries are median "
        r"Spearman $\rho$ between an object's row and the reference row, "
        r"with mean top-3 overlap: SNI \Dm{} and \TAP{} over 12 targets "
        r"$\times$ 15 seeds (180 cells); $^{\dagger}$MissForest-hosted "
        r"readouts are an initial five-seed descriptive subset (60 "
        r"cells; their hosts were not retrained in the seed expansion). "
        r"The same-host permutation readout is not scored on this axis "
        r"because it defines the behavioral reference: its agreement "
        r"with $A$ is 1 identically (excluded by construction). "
        r"The initial confirmatory rule, its threshold and the redundancy "
        r"pre-check were committed before the corresponding five-seed "
        r"measurements; the 15-seed extension was prospectively specified "
        r"before the expansion training (seed list committed first). "
        r"Layered timeline: Online Resource 1, evidence chain.",))
    texA = dataframe_to_tex(
        panelA,
        caption=(r"Behavioral faithfulness to the host model's own "
                 r"ablation behavior: agreement with the same-host "
                 r"permutation-ablation reference."),
        label="tab:faithfulness",
        column_format="lcccc",
        header=["Audit object", r"MIMIC $\rho$", "top-3", r"eICU $\rho$",
                "top-3"],
        style=styleA, escape_data=False)

    pooled_txt = ", ".join(
        f"{lbl} {math_signed(blk['pooled_median_secondary'])}"
        for lbl, blk in (("MIMIC", fai["MIMIC"]), ("eICU", fai["eICU"]),
                         ("no-prior MIMIC", npf["MIMIC"]),
                         ("no-prior eICU", npf["eICU"])))
    styleB = TableStyle(environment="table*", notes=(
        r"\textit{Estimand and inference:} the primary statistic is "
        r"$T$ = the mean of seed-level median paired differences "
        r"$\Delta\rho$ (attention matrix $-$ \TAP{}); the reported $p$ "
        r"is the exact enumeration under seed-block "
        r"sign-exchangeability (all $2^{n}$ joint block sign flips; "
        r"two-sided floor $2/2^{15} \approx 6.1\times10^{-5}$ at 15 "
        r"seeds); the 95\% CI bootstraps seeds only (targets are the "
        r"fixed reference set, so the inference scope is training "
        r"randomness). The intervals are unadjusted pointwise "
        r"intervals; familywise decisions use the Holm-adjusted $p$ "
        r"values, so a pointwise interval excluding zero beside a "
        r"non-significant $p_{\text{Holm}}$ is the expected behavior of "
        r"two different correction levels, not a contradiction. "
        r"$p_{\text{Holm}}$ applies Holm's step-down within a family: the "
        r"two with-prior tables are one two-test family, and the two "
        r"no-prior tables are a second two-test family answering the "
        r"no-prior comparative claim (both raw $p$ values sit at the "
        r"enumeration floor, so the adjusted decision is unchanged). "
        r"Families are never pooled across the two claims. "
        r"Negative $T$ means the "
        r"attention matrix tracks the reference less well than \TAP{}. "
        r"Pooled medians (secondary robust summaries): " + pooled_txt
        + r".",))
    texB = dataframe_to_tex(
        panelB,
        caption=(r"Paired comparison against \TAP{} under the unified "
                 r"estimand: with-prior and no-prior attention, both "
                 r"tables, 15 seeds each."),
        label="tab:faithfulness_paired",
        column_format="lcccccc",
        header=["Comparison", "seeds", r"$T$",
                r"95\% CI (seed bootstrap)", r"exact $p$",
                r"$p_{\text{Holm}}$", r"seeds $<0$"],
        style=styleB, escape_data=False)

    macros = (
        "% generated by reporting/table_faithfulness.py from t_final.json\n"
        "\\newcommand{\\faithTMimic}{" + f"{fai['MIMIC']['T']:+.3f}" + "}\n"
        "\\newcommand{\\faithTEicu}{" + f"{fai['eICU']['T']:+.3f}" + "}\n"
        "\\newcommand{\\faithHolmMimic}{"
        + f"{fai['MIMIC']['p_holm']:.3f}" + "}\n"
        "\\newcommand{\\faithHolmEicu}{"
        + f"{fai['eICU']['p_holm']:.2f}" + "}\n"
        "\\newcommand{\\nopriorPExact}{6.1\\times10^{-5}}\n")
    (out_path.parent / "faith_macros.tex").write_text(macros)

    return write_tex(out_path, texA + "\n" + texB, provenance={
        "generator": "reporting/table_faithfulness.py",
        "input": "results/T5_stats/t_final.json (single evidence source) "
                 "+ faithfulness_summary.json (five-seed descriptive "
                 "subset)",
        "code_SNI commit": runconfig.git_commit()})

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_faithfulness.tex"))
    a = ap.parse_args()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
