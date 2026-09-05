"""T1.3 equivalence comparison: ported code_SNI vs the frozen R0 results.

Reads the runs produced by ``equivalence_run.py`` and the reference results in
``project_sni_R0/results_all/sni_v03_main/``, and applies the acceptance rule
from the P1 instruction:

    every metric's deviation must be smaller than the cross-seed standard
    deviation of that metric on the same dataset

The cross-seed spread of R0's own five seeds is the natural yardstick: it is the
noise floor below which "the port changed something" is not distinguishable from
"this method is stochastic".

Three deviations are reported separately, because they answer different questions:

  per-seed        does seed s of the port match seed s of R0?
                  Expected to FAIL by construction — see B48: R0's per-feature
                  seed is ``cfg.seed + hash(feature) % 10000`` and CPython
                  randomises str hashing per process, so R0's own seed s is not
                  reproducible even by R0.
  mean-vs-mean    does the 5-seed mean of the port match the 5-seed mean of R0?
                  This is the meaningful test of faithfulness.
  within-condition does the port reproduce ITSELF when re-run?
                  Requires --repeat runs; quantifies how much of any gap is
                  irreducible stochasticity.

Usage
-----
    PYTHONPATH=$PWD \
    python code_SNI/tests/equivalence_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
R0_RESULTS = CODE_ROOT.parent / "project_sni_R0" / "results_all" / "sni_v03_main"
EQ = CODE_ROOT / "results" / "T1.3_equivalence"

METRICS = ["cont_NRMSE", "cont_RMSE", "cont_MAE", "cont_R2", "cont_Spearman",
           "cat_Accuracy", "cat_Macro-F1", "cat_Cohen_kappa", "lambda_mean"]
DATASETS = ["MIMIC", "NHANES"]
SEEDS = [1, 2, 3, 5, 8]


def load_r0() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            p = R0_RESULTS / f"V03_MAIN_{ds}_MAR_30per_SNI_s{s}" / "metrics_summary.json"
            j = json.loads(p.read_text())
            j.update(dataset=ds, seed=s, source="R0")
            rows.append(j)
    return pd.DataFrame(rows)


def load_port(condition: str) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            p = EQ / condition / f"{ds}_MAR_30per_SNI_s{s}" / "metrics_summary.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            j.update(source=condition)
            rows.append(j)
    return pd.DataFrame(rows)


def frobenius_rel(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Relative Frobenius distance between two dependency matrices."""
    common = [c for c in a.columns if c in b.columns]
    A = a.loc[common, common].to_numpy(dtype=float)
    B = b.loc[common, common].to_numpy(dtype=float)
    den = np.linalg.norm(A)
    return float(np.linalg.norm(A - B) / den) if den > 0 else float("nan")


def d_matrix_comparison(condition: str) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            pr = EQ / condition / f"{ds}_MAR_30per_SNI_s{s}" / "dependency_matrix.csv"
            r0 = R0_RESULTS / f"V03_MAIN_{ds}_MAR_30per_SNI_s{s}" / "dependency_matrix.csv"
            if not pr.exists():
                continue
            A = pd.read_csv(r0, index_col=0)
            B = pd.read_csv(pr, index_col=0)
            common = [c for c in A.columns if c in B.columns]
            a = A.loc[common, common].to_numpy(float)
            b = B.loc[common, common].to_numpy(float)
            off = ~np.eye(len(common), dtype=bool)
            rows.append(dict(
                dataset=ds, seed=s, condition=condition,
                frobenius_rel=frobenius_rel(A, B),
                spearman_offdiag=float(pd.Series(a[off]).corr(pd.Series(b[off]), method="spearman")),
                max_abs_cell_diff=float(np.abs(a - b).max()),
            ))
    return pd.DataFrame(rows)


def main() -> int:
    r0 = load_r0()
    EQ.mkdir(parents=True, exist_ok=True)

    # Yardstick: R0's own cross-seed spread, per dataset per metric.
    yard = r0.groupby("dataset")[METRICS].std(ddof=1)
    yard.to_csv(EQ / "yardstick_r0_cross_seed_std.csv")
    print("=== Yardstick: R0 cross-seed std (n=5) ===")
    print(yard.to_string(float_format=lambda v: f"{v:.6f}"))

    all_rows = []
    for condition in ["deterministic", "r0_performance"]:
        port = load_port(condition)
        if port.empty:
            print(f"\n[skip] no runs for condition {condition}")
            continue

        print(f"\n\n{'='*78}\nCONDITION: {condition}   ({len(port)} runs)\n{'='*78}")

        # --- per-seed deviation -------------------------------------------
        m = port.merge(r0, on=["dataset", "seed"], suffixes=("_port", "_r0"))
        print("\n--- per-seed |port - R0| vs cross-seed std ---")
        for ds in DATASETS:
            sub = m[m.dataset == ds]
            if sub.empty:
                continue
            print(f"\n  {ds}")
            print(f"    {'metric':<18} {'R0 mean':>11} {'port mean':>11} "
                  f"{'max|diff|':>11} {'std(R0)':>11} {'ratio':>7}")
            for met in METRICS:
                d = (sub[f"{met}_port"] - sub[f"{met}_r0"]).abs()
                sd = yard.loc[ds, met]
                ratio = d.max() / sd if sd > 0 else np.nan
                print(f"    {met:<18} {sub[f'{met}_r0'].mean():>11.6f} "
                      f"{sub[f'{met}_port'].mean():>11.6f} {d.max():>11.6f} "
                      f"{sd:>11.6f} {ratio:>7.2f}")

        # --- mean-vs-mean deviation (the meaningful test) -----------------
        #
        # The P1 instruction's criterion is "deviation smaller than the
        # cross-seed standard deviation", applied here as `ratio_sd`.
        #
        # A second column is reported alongside it because the literal criterion
        # is conservative in one direction and lenient in another. Both the R0
        # mean and the port mean are averages of five runs, and B48 means the
        # five runs on each side differ by uncontrolled per-feature seeds. The
        # difference of two independent 5-run means therefore has standard
        # deviation sigma*sqrt(2/5) = 0.63*sigma, not sigma. `ratio_se` divides
        # by that instead, so a value below about 2 is what "indistinguishable
        # from run-to-run noise" actually means. Both are reported; the verdict
        # column follows the instruction's criterion.
        print("\n--- 5-seed mean deviation vs cross-seed std ---")
        for ds in DATASETS:
            sub = m[m.dataset == ds]
            if sub.empty:
                continue
            n_port = len(sub)
            print(f"\n  {ds}  (n_port={n_port})")
            print(f"    {'metric':<18} {'R0 mean':>11} {'port mean':>11} "
                  f"{'|diff|':>11} {'std(R0)':>11} {'ratio_sd':>9} {'ratio_se':>9}  verdict")
            for met in METRICS:
                dm = abs(sub[f"{met}_port"].mean() - sub[f"{met}_r0"].mean())
                sd = yard.loc[ds, met]
                ratio = dm / sd if sd > 0 else np.nan
                se = sd * np.sqrt(2.0 / max(n_port, 1)) if sd > 0 else np.nan
                ratio_se = dm / se if se and se > 0 else np.nan
                verdict = "PASS" if ratio < 1.0 else "FAIL"
                print(f"    {met:<18} {sub[f'{met}_r0'].mean():>11.6f} "
                      f"{sub[f'{met}_port'].mean():>11.6f} {dm:>11.6f} "
                      f"{sd:>11.6f} {ratio:>9.2f} {ratio_se:>9.2f}  {verdict}")
                all_rows.append(dict(condition=condition, dataset=ds, metric=met,
                                     n_port=n_port,
                                     r0_mean=sub[f"{met}_r0"].mean(),
                                     port_mean=sub[f"{met}_port"].mean(),
                                     abs_mean_diff=dm, r0_cross_seed_std=sd,
                                     ratio_sd=ratio, ratio_se=ratio_se, verdict=verdict,
                                     max_per_seed_diff=(sub[f"{met}_port"] - sub[f"{met}_r0"]).abs().max()))

        # --- dispersion comparison ----------------------------------------
        #
        # Under B48 neither side's individual seeds are reproducible, so matching
        # them one-to-one is impossible in principle. What a faithful port CAN be
        # asked to match is the DISTRIBUTION: if the ported code has the same
        # run-to-run spread as R0, the two are drawing from the same generative
        # process, and a per-seed gap is noise rather than a defect. A port that
        # had introduced a bug would typically show a visibly different spread.
        if len(sub) >= 3:
            print("\n--- cross-seed dispersion: port vs R0 (same process, or not?) ---")
            for ds in DATASETS:
                s = m[m.dataset == ds]
                if len(s) < 3:
                    continue
                print(f"\n  {ds}  (n_port={len(s)})")
                print(f"    {'metric':<18} {'sd(R0)':>11} {'sd(port)':>11} "
                      f"{'ratio':>8}   interpretation")
                for met in METRICS:
                    sd_r0 = s[f"{met}_r0"].std(ddof=1)
                    sd_pt = s[f"{met}_port"].std(ddof=1)
                    r = sd_pt / sd_r0 if sd_r0 > 0 else np.nan
                    note = ("comparable" if 0.4 <= r <= 2.5 else
                            "port much tighter" if r < 0.4 else "port much wider")
                    print(f"    {met:<18} {sd_r0:>11.6f} {sd_pt:>11.6f} "
                          f"{r:>8.2f}   {note}")

        # --- dependency matrix --------------------------------------------
        dcmp = d_matrix_comparison(condition)
        if not dcmp.empty:
            print("\n--- dependency matrix D vs R0 ---")
            print(dcmp.groupby("dataset")[["frobenius_rel", "spearman_offdiag", "max_abs_cell_diff"]]
                  .agg(["mean", "min", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))
            dcmp.to_csv(EQ / f"d_matrix_comparison_{condition}.csv", index=False)

    res = pd.DataFrame(all_rows)
    if not res.empty:
        res.to_csv(EQ / "equivalence_verdict.csv", index=False)
        print(f"\n\n{'='*78}\nOVERALL\n{'='*78}")
        for condition in res.condition.unique():
            sub = res[res.condition == condition]
            n_fail = int((sub.verdict == "FAIL").sum())
            cont = sub[sub.metric.str.startswith("cont_")]
            cat = sub[sub.metric.str.startswith("cat_")]
            print(f"  {condition:<18} {len(sub) - n_fail}/{len(sub)} metric-dataset "
                  f"combinations within the cross-seed noise floor"
                  f"{'  -> PASS' if n_fail == 0 else f'  -> {n_fail} FAIL'}")
            print(f"    {'continuous metrics':<22} "
                  f"{int((cont.verdict=='PASS').sum())}/{len(cont)} pass, "
                  f"max ratio_sd {cont.ratio_sd.max():.2f}")
            print(f"    {'categorical metrics':<22} "
                  f"{int((cat.verdict=='PASS').sum())}/{len(cat)} pass, "
                  f"max ratio_sd {cat.ratio_sd.max():.2f}")
            print(f"    {'by the ratio_se yardstick':<22} "
                  f"{int((sub.ratio_se < 2).sum())}/{len(sub)} below 2")
        print(f"\n  detail -> {EQ / 'equivalence_verdict.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
