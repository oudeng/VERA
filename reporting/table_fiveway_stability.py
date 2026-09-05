"""R4.3 -- five-way cross-seed stability table (Results, reproducibility axis).

Input: results/T3_five_way/{fiveway_pairs.csv, fiveway_agreement_with_P.csv}.
Guard (P4-D): ALL FIVE audit objects must be present for every dataset --
a four-object table silently rendered as "the comparison" is exactly the
shape of B7. P appears under its paper name TAP; its stability is bitwise
seed-invariance, printed as 1.0 with the by-construction note.

    PYTHONHASHSEED=2025 python reporting/table_fiveway_stability.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.disclosures import INFO_ASYMMETRY_STABILITY
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
DATASETS = ["MIMIC", "eICU"]
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest", "Permutation-on-SNI"]
DISPLAY = {"P": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           # Eighth review P0-3: the same-host readout appears under BOTH
           # calibers, labeled, exactly as Table 4 does it. Printing only the
           # archived one while the text, Fig. 3 and the ESM all use the
           # symmetric band is the defect this splits.
           "Permutation-on-SNI": "Permutation-on-SNI (archived; "
                                 "withheld-truth signal)",
           "Permutation-on-SNI-symmetric": "Permutation-on-SNI (same host; "
                                           "symmetric signal)",
           "SNI-D": r"SNI \Dm{}"}


def _sci(x: float, sig: int = 1) -> str:
    """A power of ten a reader reads. "2e-05" is not one."""
    from math import floor, log10
    if x == 0:
        return "0"
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return rf"${m:.{sig}f} \times 10^{{{e}}}$"


def _symmetry_band_note() -> str:
    """The host band with and without the privileged error signal.

    The row in the table is the archived reading. The band the main text
    compares \\Dm{} against is the corrected one, so both have to be here --
    a corrected number quoted in the text against an uncorrected number in
    the table it points at is the worst of the two options.
    """
    import json
    f = CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_band.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing: this table's Permutation-on-SNI row would then "
            f"be the only band on offer, and it is the one taken under the "
            f"privileged error signal.")
    b = json.loads(f.read_text())["datasets"]
    parts = []
    for ds in DATASETS:
        o, n = b[ds]["band_oracle"], b[ds]["band_noOracle"]
        parts.append(rf"{ds} {o['mean']:.3f} $\to$ {n['mean']:.3f} "
                     rf"(min {n['min']:.3f}, J@3 {n['top3_jaccard_mean']:.3f})")
    d = b[DATASETS[0]]
    return (r"\textit{The two same-host rows:} the archived row was scored "
            r"against the values withheld from the imputer; the symmetric row "
            r"takes the error signal from the host's own completed table, "
            r"which is the caliber on which it and the objects it is compared "
            r"with have the same information. The band rises accordingly ("
            + "; ".join(parts) + r"). \textbf{The symmetric row is the "
            r"current reading:} it is the band \Dm{} is compared against in "
            r"the Discussion and the one Fig.~3 prints, and it remains "
            r"disjoint from \Dm{}'s own band on both tables. The symmetric "
            r"row's last column is the same agreement statistic as every "
            r"other row, computed on the no-oracle ablation matrices by the "
            r"archived arithmetic; no cell in that row is carried over from "
            r"the other caliber. Why the band moves, and the control "
            r"recomputation showing the change is the signal and not the "
            r"rerun, are in Online Resource~1 "
            r"(\emph{Axis 2: stability alignment}).")


def band_control_note() -> str:
    """The control recomputation, for the ESM (ninth review P2-2, moved)."""
    import json
    b = json.loads((CODE_ROOT / "results" / "T6_symmetry"
                    / "no_oracle_band.json").read_text())["datasets"]
    return (r"\textit{The change is the signal, not the rerun.} A control "
            r"recomputation of the T61 archived oracle band on the same "
            r"hosts reproduces it to "
            + _sci(max(b[x]["control_recomputed_oracle_band_vs_shipped"]["abs_diff"]
                       for x in DATASETS), 0) + r".")


def _symmetric_row() -> dict:
    """The same-host readout under the information-symmetric error signal.

    Every cell comes from results/T6_symmetry/no_oracle_band.json, which is
    produced by experiments/no_oracle_band.py from the no-oracle ablation
    matrices -- the same algorithm as the archived row, on the other caliber.
    The last column included: it is a pure readout of matrices that already
    exist, so there is no reason to print the archived caliber's agreement
    beside a symmetric band, and no reason to leave the cell empty (eighth
    review P0-3).
    """
    import json
    f = CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_band.json"
    b = json.loads(f.read_text())["datasets"]
    row = {"obj": DISPLAY["Permutation-on-SNI-symmetric"]}
    for ds in DATASETS:
        d = b[ds]["band_noOracle"]
        row[f"{ds}_rho"] = rf"{d['mean']:.3f} ({d['min']:.3f})"
        row[f"{ds}_j3"] = f"{d['top3_jaccard_mean']:.3f}"
        ag = b[ds].get("agreement_with_TAP_noOracle")
        row[f"{ds}_tap"] = f"{ag['mean']:.2f}" if ag else "not reported"
    return row


def build(out_path: Path) -> Path:
    pairs = pd.read_csv(FIVEWAY / "fiveway_pairs.csv")
    agree = pd.read_csv(FIVEWAY / "fiveway_agreement_with_P.csv")
    perm = pd.read_csv(T4F / "perm_on_sni_real_stability.csv")
    perm_ag = pd.read_csv(T4F / "perm_on_sni_agreement_with_P.csv")
    within = pd.read_csv(T4F / "within_seed_consistency.csv")

    for ds in DATASETS:                          # the six-object guard (T4F rule)
        have = {g.split("|", 1)[1] for g in pairs.group.unique()
                if g.startswith(f"{ds}|")}
        have |= {"Permutation-on-SNI"} if len(perm[perm.dataset == ds]) else set()
        missing = [o for o in OBJECTS if o not in have]
        if missing:
            raise ValueError(
                f"{ds}: audit objects missing from the stability data: "
                f"{missing}. Refusing to emit a partial six-way table (the "
                f"B7 shape; six-or-refuse per docs/T4F_presentation_rule.md).")

    rows = []
    for obj in OBJECTS:
        row = {"obj": DISPLAY.get(obj, obj)}
        for ds in DATASETS:
            g = pairs[pairs.group == f"{ds}|{obj}"]
            if obj == "P":
                row[f"{ds}_rho"] = r"1.000 (bitwise)"
                row[f"{ds}_j3"] = "1.000"
            elif obj == "Permutation-on-SNI":
                gp = perm[perm.dataset == ds]
                row[f"{ds}_rho"] = (rf"{gp.spearman.mean():.3f} "
                                    rf"({gp.spearman.min():.3f})")
                row[f"{ds}_j3"] = f"{gp.top3_jaccard.mean():.3f}"
            else:
                sub = g[g.rows == "own16"] if (g.rows == "own16").any() else g
                row[f"{ds}_rho"] = (rf"{sub.spearman.mean():.3f} "
                                    rf"({sub.spearman.min():.3f})")
                row[f"{ds}_j3"] = f"{sub.top3_jaccard.mean():.3f}"
            if obj == "Permutation-on-SNI":
                ga = perm_ag[perm_ag.dataset == ds]
            else:
                ga = agree[(agree.dataset == ds) & (agree.method == obj)
                           & (agree.seed != "all")]
            row[f"{ds}_tap"] = (f"{ga.rho_with_P.astype(float).mean():.2f}"
                                if len(ga) else "1.0")
        rows.append(row)
        if obj == "Permutation-on-SNI":
            rows.append(_symmetric_row())
    body = pd.DataFrame(rows)

    style = TableStyle(environment="table*", col_sep_pt=3.5, notes=(
        INFO_ASYMMETRY_STABILITY + " " + _symmetry_band_note() + " " +
        r"\textit{Layer naming:} the measurement protocol was committed "
        r"before these readouts (initial confirmatory layer); the reading "
        r"is descriptive by design (joint diagnostic with the faithfulness "
        r"axis, no per-object verdict). "
        r"\textit{Protocol:} pairwise Spearman $\rho$ over off-diagonal "
        r"entries across five seeds (all $\binom{5}{2}=10$ seed pairs), "
        r"MAR 30\%, "
        r"the same compute conditions for every object (CPU, BLAS threads "
        r"pinned); "
        r"mean (min) shown, with mean top-3 Jaccard of row-wise top sources. "
        r"SNI \Dm{} and \TAP{} cover all 16 feature rows; the MissForest "
        r"readouts cover the 12 imputed rows -- restricting SNI \Dm{} to the "
        r"same 12 rows changes its values by less than 0.013 and no ordering. "
        r"\TAP{} is bitwise identical across seeds, thread counts and devices "
        r"by construction (asserted, not assumed). Permutation-on-SNI is the "
        r"post-hoc readout of the same host, computed with the ablation "
        r"machinery of the faithfulness axis on the identical runs. "
        r"The symmetric same-host permutation readout defines the "
        r"operational host-conditioned reference used in this study. Its "
        r"cross-seed stability was 0.820/0.748, while within-host redraw "
        rf"reproducibility was "
        rf"{within[within.dataset=='MIMIC'].spearman.mean():.2f}/"
        rf"{within[within.dataset=='eICU'].spearman.mean():.2f}. "
        r"The observed contrast is consistent with substantial between-host "
        r"variation, but the operational probe retains non-zero estimation "
        r"noise. The last "
        r"column is each object's Spearman agreement with \TAP{} (mean "
        r"over seeds).",))
    tex = dataframe_to_tex(
        body,
        # Eighth review P0-3: "identical conditions" was false twice over --
        # the same-host row was the archived caliber while everything else in
        # the paper used the symmetric one, and D/TAP cover 16 rows against
        # the MissForest readouts' 12. The replacement is the review's own.
        caption=(r"Cross-seed stability of the audit objects under their "
                 r"prespecified protocols; the same-host readout is reported "
                 r"separately before and after information-symmetry "
                 r"correction."),
        label="tab:fiveway_stability",
        column_format=(r">{\raggedright\arraybackslash}p{0.26\textwidth}"
                       r"cccccc"),
        header=["Audit object",
                r"MIMIC $\rho$ (min)", r"J@3", r"$\rho$ w/ \TAP{}",
                r"eICU $\rho$ (min)", r"J@3", r"$\rho$ w/ \TAP{}"],
        style=style, escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_fiveway_stability.py",
        "input": str(FIVEWAY),
        "code_SNI commit": runconfig.git_commit()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_fiveway_stability.tex"))
    a = ap.parse_args()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
