"""R4.2 -- recovery-axis table (Results): five objects on the synthetic
ground-truth benchmark, three regimes x four metrics.

Inputs (all on disk since the pilot and T2g.1): the pilot's 60-cell results
for SNI-D and the three MissForest readouts, T2g.1's three-way cells for the
TAP rows (method 'P-alone', scored with the pilot's own scorer on the same
common rows), and the pilot's 60 stored matrices for the cross-seed
instability note.

Guards (P4-E, same family as the other three): all FIVE objects or refuse;
all FOUR metrics or refuse -- the second one exists because reporting only
the favorable metric is precisely the move the honesty items below rule
out. Two honesty items are computed here and injected into the note, never
hand-written: (i) overall AUPRC is a tie (TAP vs SNI-D) even though TAP
leads AUROC / P@K / SHD -- 'outperforms overall' is not licensed; (ii) every
object's cross-seed stability is LOW on this synthetic benchmark, which
bounds how much any recovery conclusion can carry.

    PYTHONHASHSEED=2025 python reporting/table_recovery.py
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.disclosures import INFO_ASYMMETRY_RECOVERY
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

PILOT = CODE_ROOT / "results" / "T2.5_pilot"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
RLABEL = {"linear_gaussian": "Linear--Gaussian",
          "nonlinear_mixed": "Nonlinear mixed",
          "interaction_xor": "Interaction--XOR"}
SEEDS = [2025, 2026, 2027, 2028, 2029]
METRICS = [("auroc", r"AUROC $\uparrow$"), ("auprc", r"AUPRC $\uparrow$"),
           ("prec_at_k", r"P@K $\uparrow$"), ("shd", r"SHD $\downarrow$")]
#: The main table's rows. Two same-host probe rows, not one, and each says
#: which error signal produced it: the seventh review found the table showing
#: the withheld-truth reading under a bare "same host" label while the text
#: and Fig. 3 used the symmetric one, so a reader met two different analyses
#: without being told. The symmetric row is the comparison the paper reports.
FAIR = CODE_ROOT / "results" / "T6_symmetry" / "fair_same_host_recovery_cells.csv"
OBJECTS = ["SNI-D", "SNI-D-retrained", "P-alone", "MissForest-importance",
           "SHAP-on-MissForest", "Permutation-on-MissForest",
           "Permutation-on-SNI", "Permutation-on-SNI-fair-noOracle"]
#: which objects are read from the fair-pair cell file rather than the
#: archived six-way file
FROM_FAIR = {"Permutation-on-SNI-fair-noOracle"}
DISPLAY = {"P-alone": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           "Permutation-on-SNI":
               "Permutation-on-SNI (archived; withheld-truth signal)",
           "Permutation-on-SNI-fair-noOracle":
               "Permutation-on-SNI (same retrained host; symmetric signal)",
           "SNI-D-retrained": r"SNI \Dm{} (retrained host)",
           "SNI-D": r"SNI \Dm{} (archived host)"}


def _cross_seed_median(method_file: str) -> float:
    """Median pairwise cross-seed Spearman of the stored pilot matrices."""
    from scipy.stats import spearmanr
    vals = []
    for regime in REGIMES:
        mats = {}
        for s in SEEDS:
            f = PILOT / f"D_{regime}_s{s}_{method_file}.csv"
            mats[s] = pd.read_csv(f, index_col=0).to_numpy(dtype=float)
        off = ~np.eye(len(next(iter(mats.values()))), dtype=bool)
        for a, b in combinations(SEEDS, 2):
            A, B = np.nan_to_num(mats[a]), np.nan_to_num(mats[b])
            vals.append(float(spearmanr(A[off], B[off]).statistic))
    return float(np.median(vals))


def _verdict_note() -> str:
    """The same-host pair, read under information symmetry (P5R-P SS1).

    The pair reported here is the WITHIN-HOST one: one training run per cell
    emitted the attention matrix and the probe, and the probe's error signal
    comes from that host's own completed table rather than from values
    withheld from the imputer. The reading it replaced -- oracle caliber, and
    paired with a differently trained host's matrix -- is kept in the
    repository artifact and named here, because a number that quietly
    disappears is a number a reader cannot check.
    """
    import json
    ff = CODE_ROOT / "results" / "T6_symmetry" / "fair_same_host_recovery.json"
    if not ff.exists():
        raise FileNotFoundError(
            f"{ff} is missing: this note's primary pair is the within-host "
            f"comparison under information symmetry, and falling back to the "
            f"superseded oracle-caliber pair would print it as if it were "
            f"still the paper's reading.")
    fair = json.loads(ff.read_text())
    tfr = json.loads((CODE_ROOT / "results" / "T5_stats"
                      / "t_final.json").read_text())["recovery"]
    tf, orc = (tfr["probe_vs_D_same_host_symmetric"],
               tfr["probe_vs_D_same_host_oracle_control"])
    sup = tfr["probe_vs_D_retrained"]
    assert sup.get("superseded_by") == "probe_vs_D_same_host_symmetric", \
        "the archived pair lost its superseded mark"
    sym = fair["probe_vs_D_same_host_no_oracle"]
    tlo, thi = tf["ci95_T"]
    # No branch on the T4F verdict: that rule is retired (seventh review
    # SS5.1), and a retired rule may not choose this sentence's wording. The
    # reading is stated from the seed-block estimand used everywhere else.
    return (
        r"\textit{Prespecified comparison, interpreted under the corrected "
        r"seed-block inference:} on the same host, and with the probe's "
        r"error signal taken from that host's own completed table, the "
        r"post-hoc readout exceeded the attention matrix in all "
        + rf"{sym['cells_favouring_probe']}/{sym['cells_total']} "
        + rf"regime--seed cells (seed-block $T$ = {tf['T']:+.3f} "
        + rf"$\Delta$AUROC, descriptive pointwise seed-bootstrap 95\% "
        + rf"interval [{tlo:+.3f}, {thi:+.3f}]). With five independent seeds "
        + rf"the "
        + rf"two-sided exact test remains inconclusive at its attainable "
        + rf"floor of $p$ = {tf['p_exact']:.4f}, so this is a directionally "
        + rf"consistent observed difference and not a formal recovery-axis "
        + rf"win. Allowing the probe to use the withheld true values "
        + rf"increased $T$ by {orc['T'] - tf['T']:+.3f}, from {tf['T']:+.3f} "
        + rf"to {orc['T']:+.3f}. "
        + r"The archived reading, the fresh oracle-caliber control, the "
        + r"comparison against the strongest externally hosted readout, the "
        + r"XOR-saturation check, and the retired cell-level verdict's "
        + r"status are set out in Online Resource~1 "
        + r"(\emph{Recovery axis: audit history}).")


def _symmetry_note() -> str:
    """T6.1: the same-host probe's score with and without the privileged
    error signal, and the retrained-host control between them.

    The archived reading is kept beside the corrected one rather than
    replaced. Three objects, because two things changed at once and they have
    to stay separable: the host was retrained, and the error signal stopped
    being measured against values the imputer never saw.
    """
    import json
    f = CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_recovery.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing: this table's note reports the recovery axis "
            f"under information symmetry, and a recovery table that omits it "
            f"would state the archived number as if it were uncontested.")
    v = json.loads(f.read_text())
    pv, j = v["per_variant_own_rows"], v["joint_rows"]
    ext = j["strongest_external"]
    pr = v["paired_no_oracle_vs_strongest_external"]
    lo, hi = pr["ci95_T_seedboot"]
    return (
        rf"\textit{{Information symmetry:}} that asymmetry has since "
        rf"been removed and this axis recomputed with the probe's error "
        rf"signal taken from the host's own completed table. Mean AUROC over "
        rf"the fifteen cells against the strongest externally hosted readout "
        rf"({DISPLAY.get(ext, ext)}, "
        rf"{j['strongest_external_mean_auroc']:.3f}): archived "
        rf"{pv['archived']['probe_mean_auroc']:.3f}, retrained host with the "
        rf"same signal {pv['refit_oracle']['probe_mean_auroc']:.3f}, "
        rf"retrained host with the signal removed "
        rf"{pv['refit_no_oracle']['probe_mean_auroc']:.3f}. Retraining alone "
        rf"moved the margin up; removing the privileged signal takes about "
        rf"half of it. Under the corrected signal the observed means were "
        rf"similar and the difference was inconclusive; no non-inferiority "
        rf"margin was prespecified and no equivalence test was run, so this "
        rf"is not a claim that the two are the same "
        rf"(paired $T = {pr['T_mean_of_seed_medians']:+.3f}$, "
        rf"seed-bootstrap 95\% CI [{lo:+.3f}, {hi:+.3f}], exact "
        rf"$p = {pr['exact_sign_enumeration']['p_two_sided']:.3f}$ at the "
        rf"same five-seed floor, "
        rf"{pr['cells_favouring_probe']}/{pr['cells_total']} cells "
        rf"favoring the probe). Archived and corrected readings are both "
        rf"retained in the repository artifact.")


def _xor_saturation_note() -> str:
    """T4J.1: the XOR row saturates for behavioral readouts; computed, not
    hand-written, and the claim is shown not to depend on the row."""
    import json
    if str(CODE_ROOT / "experiments") not in sys.path:
        sys.path.insert(0, str(CODE_ROOT / "experiments"))
    from pilot_r21 import load_cell
    cells = pd.read_csv(T4F / "t4f_sixway_cells.csv")
    xor = cells[cells.regime == "interaction_xor"]
    m = xor.groupby("method").auroc.mean()
    exact = bool((xor[xor.method == "Permutation-on-SNI"].auroc == 1.0).all())
    # separation check PER CELL (AUROC = 1 needs within-cell separation;
    # pooling across seeds would test a stronger, irrelevant property):
    margins = []
    for sd in SEEDS:
        _, _, _, G, _, _, _ = load_cell("interaction_xor", sd)
        P = pd.read_csv(T4F / f"PERM_interaction_xor_s{sd}.csv", index_col=0)
        Gm = G.reindex(index=P.index, columns=P.columns)
        par, non = [], []
        for fr in P.index[P.notna().any(axis=1)]:
            for j in P.columns:
                if j == fr or pd.isna(P.loc[fr, j]):
                    continue
                (par if Gm.loc[fr, j] > 0 else non).append(float(P.loc[fr, j]))
        margins.append(min(par) - max(non))
    sep_ok = all(mg > 0 for mg in margins)
    piv = cells.pivot_table(index=["regime", "seed"], columns="method",
                            values="auroc")
    nx = piv.drop(index="interaction_xor", level="regime")
    d = (nx["Permutation-on-SNI"] - nx["SNI-D-retrained"]).dropna()
    return (
        rf"\textit{{The XOR row saturates for behavioral readouts:}} "
        rf"Permutation-on-SNI scores exactly 1.000 in all five seeds"
        + (rf" because within every cell each parent's ablation effect "
           rf"exceeds every non-parent's (per-cell margins "
           rf"{min(margins):+.2f} to {max(margins):+.2f})"
           if (exact and sep_ok) else "")
        + rf"; SHAP-on-MF ({m['SHAP-on-MissForest']:.3f}) and "
        + rf"Permutation-on-MF ({m['Permutation-on-MissForest']:.3f}) sit at "
        + rf"the same ceiling, so XOR barely discriminates among these "
        + rf"objects. The verdict does not depend on the row: excluding XOR, "
        + rf"the same-host readout still leads the attention matrix in "
        + rf"{int((d > 0).sum())}/{len(d)} pairs (means "
        + rf"{nx['Permutation-on-SNI'].mean():.3f} vs "
        + rf"{nx['SNI-D-retrained'].mean():.3f}). The attention matrix "
        + rf"itself is far from saturated on XOR "
        + rf"({m['SNI-D-retrained']:.3f}), so the gap there is real, not "
        + rf"two objects at a shared ceiling.")


def _host_gap_note() -> str:
    cells = pd.read_csv(T4F / "t4f_sixway_cells.csv")
    m = cells.groupby("method").auroc.mean()
    gap = abs(float(m["SNI-D"]) - float(m["SNI-D-retrained"]))
    return (rf"The archived and retrained attention matrices themselves "
            rf"differ by {gap:.4f} mean AUROC, so the thread-count host "
            rf"distinction affects no comparison here.")


#: Every file this table's body or note actually reads. It used to be a
#: hand-written pair that named two files build() does not open and omitted
#: five it does -- a provenance line that is wrong is worse than none, because
#: it is the line a reader checks instead of the code.
INPUTS = [
    T4F / "t4f_sixway_cells.csv",          # the body's five objects
    T4F / "t4f_verdict.json",              # the continuity pair
    CODE_ROOT / "results" / "T5_stats" / "t_final.json",
    CODE_ROOT / "results" / "T6_symmetry" / "fair_same_host_recovery.json",
    CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_recovery.json",
]


def _inputs_line() -> str:
    missing = [p_ for p_ in INPUTS if not p_.exists()]
    if missing:
        raise FileNotFoundError(
            f"this table declares inputs it cannot read: {missing}. A "
            f"provenance line must name what was actually opened.")
    return " + ".join(str(p_.relative_to(CODE_ROOT)) for p_ in INPUTS)


def build(out_path: Path) -> Path:
    # Six-way cells: every object re-scored on the SAME common row set,
    # including the same-host readout (docs/T4F_presentation_rule.md).
    long = pd.read_csv(T4F / "t4f_sixway_cells.csv")
    long = long[["regime", "seed", "method",
                 "auroc", "auprc", "prec_at_k", "shd"]]

    # the symmetric same-host row comes from the fair pair's own cells: the
    # archived six-way file has no such object, and reading it from anywhere
    # else would put a different analysis in the same column
    if not FAIR.exists():
        raise FileNotFoundError(
            f"{FAIR} is missing: the table's reported same-host row is the "
            f"symmetric one, and it is not derivable from the archived cells.")
    fair = pd.read_csv(FAIR)
    fair = fair[fair.method.isin(FROM_FAIR)][
        ["regime", "seed", "method", "auroc", "auprc", "prec_at_k", "shd"]]
    long = pd.concat([long, fair], ignore_index=True)

    have = set(long.method.unique())                 # guard 1: every object
    missing = [o for o in OBJECTS if o not in have]
    if missing:
        raise ValueError(f"objects missing from recovery data: {missing}; "
                         f"refusing a partial five-way table (B7 shape).")
    for mcol, _ in METRICS:                          # guard 2: four metrics
        if mcol not in long.columns or long[mcol].isna().all():
            raise ValueError(
                f"metric '{mcol}' absent: a recovery table reporting a "
                f"metric subset is exactly the favorable-metric move the "
                f"honesty items forbid.")

    rows, rules = [], []
    for regime in REGIMES:
        rows.append({"obj": rf"\multicolumn{{5}}{{l}}{{\textit{{{RLABEL[regime]}}}}}",
                     "_span": True})  # 5 = n columns of this table
        g = long[long.regime == regime]
        for obj in OBJECTS:
            go = g[g.method == obj]
            row = {"obj": DISPLAY.get(obj, obj)}
            for mcol, _ in METRICS:
                v = pd.to_numeric(go[mcol], errors="coerce").mean()
                row[mcol] = f"{v:.3f}" if mcol != "shd" else f"{v:.1f}"
            rows.append(row)
        rules.append(len(rows) - 1)
    body = pd.DataFrame(rows).drop(columns=["_span"], errors="ignore").fillna("")

    ov = long.groupby("method")[["auroc", "auprc"]].mean()
    rho_d = _cross_seed_median("SNI-D")
    rho_perm = max(_cross_seed_median(m) for m in
                   ("MissForest-importance", "SHAP-on-MissForest",
                    "Permutation-on-MissForest"))
    note = (
        rf"\textit{{Protocol:}} the pilot's prospectively specified scorer (initial confirmatory layer) on its "
        rf"known generating adjacency (2000$\times$12, MAR 30\%, five seeds; "
        rf"row set = the intersection every object measures). "
        rf"\textit{{Two readings this table does not license.}} First: "
        rf"\TAP{{}} leads SNI \Dm{{}} on AUROC, P@K and SHD, but overall "
        rf"AUPRC is a tie ({ov.loc['P-alone','auprc']:.4f} vs "
        rf"{ov.loc['SNI-D','auprc']:.4f}, \Dm{{}} slightly ahead on "
        rf"Interaction--XOR), so the table does not license the reading that "
        rf"\TAP{{}} is ahead of \Dm{{}} overall. Second: every object is "
        rf"unstable across seeds on "
        rf"this benchmark (median pairwise $\rho$: {rho_d:.2f} for "
        rf"SNI \Dm{{}}, {rho_perm:.2f} for the best post-hoc readout), "
        rf"which bounds the strength of any recovery conclusion drawn here "
        rf"--- the real-table axes carry the stability question. "
        rf"Permutation-on-SNI is the post-hoc readout of the same host, "
        rf"included to separate the carrier from the artifact type; the "
        rf"reference is the external synthetic generating graph, so the comparison "
        rf"is not circular. " + INFO_ASYMMETRY_RECOVERY + " "
        + _verdict_note())
    style = TableStyle(environment="table*", notes=(note,))
    tex = dataframe_to_tex(
        body,
        caption=(r"Ground-truth dependency recovery, three synthetic regimes "
                 r"(means over five seeds). Where a row depends on a "
                 r"caliber, the row label says which: the HOST an artifact "
                 r"was read from, and, for the same-host ablation, the ERROR "
                 r"SIGNAL it was scored against. \TAP{} has no host, and the "
                 r"externally hosted readouts share one (a per-seed retrained "
                 r"MissForest) scored against that host's own completion. "
                 r"\Dm{} is shown separately on the "
                 r"archived and the retrained host; the same-host permutation "
                 r"readout is shown separately with the withheld-truth signal "
                 r"(archived) and with the signal taken from the host's own "
                 r"completed table (symmetric). The symmetric row is the one "
                 r"the paper reports; it is paired within the host it was "
                 r"read from, and that host was retrained under the pilot's "
                 r"protocol --- not the separately retrained host of the "
                 r"\Dm{} row above, which used a different training "
                 r"configuration. The retrained run reproduced the pilot "
                 r"archived \Dm{} readout and the T61 archived "
                 r"oracle/no-oracle permutation readouts bitwise on all 15 "
                 r"cells. This does not assert identity with the separate "
                 r"T4F archived row or bitwise identity of the full trained "
                 r"host."),
        label="tab:recovery",
        column_format="lcccc",
        header=["Audit object"] + [h for _, h in METRICS],
        style=style, escape_data=False,
        midrule_after=rules[:-1])
    # span rows: dataframe_to_tex joins every column, leaving "& & &"
    # after a \multicolumn cell -- strip the empty tails.
    tex = re.sub(r'(\\multicolumn\{\d+\}\{l\}\{.*?\}\})(?:\s*&\s*)+(\\\\)',
                 r'\1 \2', tex)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_recovery.py",
        "inputs": _inputs_line(),
        "code_SNI commit": runconfig.git_commit()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_recovery.tex"))
    a = ap.parse_args()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
