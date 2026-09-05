"""ESM fragment: probe-2 (group permutation) full readouts + reliability
qualifiers (P5R-C SS2.2). Source: results/T5_probe2/probe2_summary.json +
probe2_qualifiers.json -- the mechanical outputs of experiments/t52_probe2.

    PYTHONHASHSEED=2025 python reporting/esm_probe2.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

P2 = CODE_ROOT / "results" / "T5_probe2"
T_FINAL = CODE_ROOT / "results" / "T5_stats" / "t_final.json"


def build(out_path: Path, summary_path: Path = T_FINAL,
          qual_path: Path = P2 / "probe2_qualifiers.json") -> Path:
    from experiments.t51_cluster_stats import seed_boot_ci_T
    data = json.loads(summary_path.read_text())
    fams = (data.get("probe2_sensitivity", {}).get("families")
            or data["families"])
    q = json.loads(qual_path.read_text())
    rows = []
    for key, f in fams.items():
        variant, ds, tau = key.split("|")
        m1 = f["m1"]
        med = {int(k): [v] for k, v in
               f["aggregates"]["seed_medians"].items()}
        lo, hi = seed_boot_ci_T(med)
        neg = sum(1 for v in f["aggregates"]["seed_medians"].values()
                  if v < 0)
        rows.append({
            "Hosts": variant.replace("NoPrior", "No-prior"),
            "Table": ds, r"$\tau$": tau.split("=")[1],
            r"$T$": f"{m1['observed_stat_mean_of_block_medians']:+.3f}",
            r"Exact $p_2$": f"{m1['p_two_sided']:.6f}".rstrip("0").rstrip("."),
            r"95\% CI": f"[{lo:+.3f}, {hi:+.3f}]",
            "Seeds $<0$": f"{neg}/{f['n_seeds']}"})
    body = pd.DataFrame(rows)
    t_fam_pre = None
    t_fam = dataframe_to_tex(
        body, caption=(r"Permutation-based robustness checks (redrawn "
                       r"draws and joint group permutation on archived "
                       r"models): family-level readout per host set, "
                       r"table and cluster threshold."),
        label="tab:esm_probe2", column_format="lllcccc",
        header=list(body.columns),
        style=TableStyle(environment="table*", notes=(
            r"$T$ = mean of seed-level median $\Delta\rho_{\mathrm{group}}$ "
            r"(D $-$ \TAP{}) -- the unified primary estimand of the "
            r"Methods; 95\% CI = seed-only bootstrap of $T$; the pooled "
            r"median remains available in the archived JSON as a "
            r"secondary robust summary. Rules prospectively specified before any "
            r"group-permutation measurement, after the primary permutation "
            r"results were known (a prospectively specified "
            r"replication/sensitivity layer; "
            r"\codeid{docs/T52\_probe\_triangulation\_rules.md}). At $\tau=0.8$ no "
            r"pooled correlation exceeds the prospectively specified threshold on "
            r"either table, so the partition is all singletons and probe 2 "
            r"is a fresh-draw near-replication of the primary permutation "
            r"analysis (a repeat-draw replication sharing its "
            r"unconditional-permutation assumption); at $\tau=0.6$ "
            r"the blood-pressure cluster (\codeid{MAP\_mmHg}, \codeid{SBP\_min}, "
            r"\codeid{DBP\_min}) is "
            r"permuted jointly. Same M1/M2 machinery as the Methods "
            r"(``Statistical analysis''); enumeration floor "
            r"$2/2^{15}\approx6.1\times10^{-5}$. Cold-loaded models are "
            r"verified per host by recomputing one stored probe-1 cell on "
            r"the original RNG streams (max deviation $\le 7\times"
            r"10^{-17}$).",)), escape_data=False)
    t_fam = t_fam.replace(r"\scriptsize",
                          r"\scriptsize\setlength{\tabcolsep}{3.5pt}", 1)

    qrows = []
    for ds, qq in q.items():
        agree = qq["rho_level_agreement_all_cells"]
        b_med = float(np.median(
            qq["B_level_agreement_seed1_spearman_per_target"]))
        qrows.append({
            "Table": ds,
            r"$\Delta$ 5 reps": f"{qq['delta_median_5rep']:+.3f}",
            r"$\Delta$ 10 reps": f"{qq['delta_median_10rep']:+.3f}",
            r"B-agr.": f"{b_med:.3f}",
            r"$\rho$-agr.\ $r$":
                f"{agree['spearman_delta_probe1_vs_probe2']:.3f}",
            r"Med.\ $|\Delta\rho|$":
                f"{agree['median_abs_diff']:.3f}"})
    qb = pd.DataFrame(qrows)
    t_qual = dataframe_to_tex(
        qb, caption=(r"Probe-2 reliability qualifiers: permutation-count "
                     r"sensitivity and probe-1/probe-2 agreement."),
        label="tab:esm_probe2_qual", column_format="lccccc",
        header=list(qb.columns),
        style=TableStyle(environment="table*", notes=(
            r"$\Delta$ 5/10 reps = seed-1 median $\Delta\rho$ at 5 vs 10 "
            r"permutation repetitions; B-agr.\ = median per-target Spearman "
            r"between probe-2 singleton deltas and the stored probe-1 "
            r"ablations (seed 1); $\rho$-agr.\ $r$ = Spearman between the "
            r"two probes' per-cell $\Delta\rho$ over all 180 cells; "
            r"Med.\ $|\Delta\rho|$ = its median absolute difference. "
            r"Doubling the permutation count deepens rather than shrinks "
            r"the deficit; the difference in strength between the "
            r"primary analysis and the redrawn replication is within this "
            r"measured probe noise. No significance branch is read from "
            r"this table: it is a prospectively specified sensitivity "
            r"layer, and the $p$ values are reported for completeness, "
            r"not as confirmatory evidence. These are robustness checks of one "
            r"permutation-based reference, not an independent second "
            r"behavioral readout.",)), escape_data=False)
    return write_tex(out_path, t_fam + "\n\n" + t_qual + "\n", provenance={
        "generator": "reporting/esm_probe2.py",
        "input": f"{summary_path} + {qual_path}",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    fam = {"families": {"SNI|MIMIC|tau=0.8": {
        "n_seeds": 15,
        "m1": {"observed_stat_mean_of_block_medians": -0.008,
               "effect_pooled_median": -0.0411, "p_two_sided": 0.003418},
        "aggregates": {"seed_medians": {str(s): -0.01 for s in range(14)}
                       | {"14": 0.02}}}}}
    qual = {"MIMIC": {
        "delta_median_5rep": -0.075, "delta_median_10rep": -0.1464,
        "B_level_agreement_seed1_spearman_per_target": [0.9, 0.95, 0.92],
        "rho_level_agreement_all_cells": {
            "n_cells": 180, "spearman_delta_probe1_vs_probe2": 0.8214,
            "median_abs_diff": 0.0571}}}
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"
        qp = Path(td) / "q.json"
        sp.write_text(json.dumps(fam))
        qp.write_text(json.dumps(qual))
        out = build(Path(td) / "t.tex", sp, qp)
        txt = out.read_text()
        check("14/15" in txt, "seed-negative count 14/15 computed from "
                              "medians")
        check("-0.008" in txt and "0.003418" in txt,
              "effect and exact p carried verbatim")
        check("0.920" in txt and "0.821" in txt and "-0.146" in txt,
              "qualifier medians: B-level 0.92, r=0.821, 10-rep -0.146")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(CODE_ROOT / "reporting" / "out" / "sec_esm_probe2.tex")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
