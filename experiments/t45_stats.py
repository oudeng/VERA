"""T4.5 -- the grid's inferential statistics, per docs/T45_statistics_rules.md.

Implements the committed scheme verbatim: per family (dataset x mechanism x
rate x metric), Friedman across the nine methods over the five seeds;
post-hoc SNI-vs-baseline Wilcoxon with Holm over the 8 pairs of the family;
rank-biserial effect sizes; seeded 10k percentile bootstrap CIs of the median
paired difference; TOST at the noise-anchored margin delta = 0.5 x
median-over-methods of the seed-SD. Exact metric keys: "cont_NRMSE" and
"cat_Macro-F1" (the rule doc's "cat_MacroF1" names the same metric; the
hyphenated form is the recorded key). Families whose metric an entire
dataset never defines are listed as undefined; a family where some method
is missing the metric while others have it is an integrity refusal.

    env PYTHONHASHSEED=2025 python experiments/t45_stats.py --stage selftest
    env PYTHONHASHSEED=2025 python experiments/t45_stats.py --stage run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

GRID = CODE_ROOT / "results" / "P2_main_grid"
OUT = CODE_ROOT / "results" / "T4_stats"

METRICS = [("cont_NRMSE", False), ("cat_Macro-F1", True)]  # (key, higher_better)
METHODS = ["MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE",
           "HyperImpute", "TabCSDI", "SNI"]
SEEDS = [1, 2, 3, 5, 8]
BOOT_SEED = 20260827
ALPHA = 0.05


def wilcoxon_p(vec: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    v = np.asarray(vec, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    if np.allclose(v, 0.0):
        return 1.0
    return float(wilcoxon(v).pvalue)


def family_matrix(df: pd.DataFrame, ds, mech, rate, metric):
    """Seeds x methods matrix for one family, or a (reason, detail) skip.
    Missing-for-all-methods -> undefined; missing for SOME -> integrity
    refusal (raise)."""
    sel = df[(df.dataset == ds) & (df.mechanism == mech)]
    sel = sel[sel.rate.isna()] if rate is None else sel[sel.rate == rate]
    if metric not in sel.columns or sel[metric].isna().all():
        return None, "metric_undefined_for_family"
    mat = sel.pivot_table(index="seed", columns="method", values=metric,
                          aggfunc="first")
    missing = [m for m in METHODS if m not in mat.columns
               or mat[m].isna().any()]
    if missing:
        raise ValueError(
            f"family ({ds},{mech},{rate},{metric}): methods with missing "
            f"values: {missing} -- integrity refusal (rules: never imputed)")
    mat = mat.loc[sorted(mat.index), METHODS]
    assert list(mat.index) == SEEDS, f"seeds {list(mat.index)} != {SEEDS}"
    return mat, None


def tost_margin(mat: pd.DataFrame) -> float:
    """delta = 0.5 x median over methods of the seed-SD (rules, fixed)."""
    return 0.5 * float(np.median(mat.std(axis=0, ddof=1)))


def analyze_family(mat: pd.DataFrame, higher_better: bool) -> dict:
    from scipy.stats import friedmanchisquare
    from stats.posthoc import holm_bonferroni
    from stats.effect_size import rank_biserial_from_diffs
    from stats.equivalence import tost_paired

    fr_stat, fr_p = friedmanchisquare(*[mat[m].to_numpy() for m in METHODS])
    delta = tost_margin(mat)
    pairs, pvals = [], []
    for m in METHODS:
        if m == "SNI":
            continue
        d = (mat["SNI"] - mat[m]).to_numpy(float)
        p = wilcoxon_p(d)
        rng = np.random.default_rng(BOOT_SEED)
        boots = np.median(rng.choice(d, size=(10_000, len(d)), replace=True),
                          axis=1)
        try:
            t = tost_paired(d, delta=delta)
            tostd = {"equivalent": bool(t.equivalent), "p_lower": float(t.p_lower),
                     "p_upper": float(t.p_upper)}
        except Exception as exc:   # degenerate vectors etc. -- recorded
            tostd = {"equivalent": None, "error": repr(exc)[:120]}
        # All-zero diffs: zero effect, not an exception/NaN -- the committed
        # semantics follow faithfulness._paired_effect.
        rb = 0.0 if np.allclose(d, 0.0) else float(rank_biserial_from_diffs(d))
        pairs.append({"baseline": m, "median_delta": float(np.median(d)),
                      "wilcoxon_p": p,
                      "rank_biserial": rb,
                      "median_ci95": [float(np.percentile(boots, 2.5)),
                                      float(np.percentile(boots, 97.5))],
                      "tost": tostd})
        pvals.append(p)
    holm = holm_bonferroni(pvals)
    for rec, hp in zip(pairs, holm):
        rec["wilcoxon_p_holm"] = float(hp)
    return {"friedman_chi2": float(fr_stat), "friedman_p": float(fr_p),
            "n_methods": len(METHODS), "n_seeds": len(SEEDS),
            "tost_margin": round(delta, 6),
            "higher_is_better": higher_better,
            "sni_vs_baselines": pairs}


def stage_run() -> int:
    from stats.aggregate_grid import load_runs
    df = load_runs(GRID)
    fams, skipped = {}, []
    keys = df[["dataset", "mechanism", "rate"]].drop_duplicates()
    for _, k in keys.iterrows():
        for metric, hb in METRICS:
            rate = None if pd.isna(k.rate) else float(k.rate)
            mat, skip = family_matrix(df, k.dataset, k.mechanism, rate, metric)
            fam_key = f"{k.dataset}|{k.mechanism}|{rate}|{metric}"
            if mat is None:
                skipped.append({"family": fam_key, "reason": skip})
                continue
            fams[fam_key] = analyze_family(mat, hb)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"rule_doc": "docs/T45_statistics_rules.md",
           "grid": str(GRID), "n_families": len(fams),
           "n_undefined": len(skipped),
           "alpha": ALPHA, "boot_seed": BOOT_SEED,
           "families": fams, "undefined_families": skipped}
    (OUT / "t45_stats.json").write_text(json.dumps(out, indent=1))
    rows = []
    for fk, f in fams.items():
        for p in f["sni_vs_baselines"]:
            rows.append({"family": fk, **{k: v for k, v in p.items()
                                          if k != "tost"},
                         **{f"tost_{k}": v for k, v in p["tost"].items()}})
    pd.DataFrame(rows).to_csv(OUT / "t45_pairs.csv", index=False)
    print(f"[ok] {len(fams)} families analyzed, {len(skipped)} undefined; "
          f"wrote {OUT / 't45_stats.json'}")
    return 0


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    rng = np.random.default_rng(1)
    # family builder: full matrix passes; partial missing refuses;
    # metric absent -> undefined skip
    base = []
    for s in SEEDS:
        for m in METHODS:
            base.append({"dataset": "D", "mechanism": "MAR", "rate": 0.3,
                         "seed": s, "method": m,
                         "cont_NRMSE": float(rng.uniform(0.1, 0.5))})
    df = pd.DataFrame(base)
    mat, skip = family_matrix(df, "D", "MAR", 0.3, "cont_NRMSE")
    check(mat is not None and mat.shape == (5, 9) and skip is None,
          "family builder: 5x9 matrix, method order fixed")
    _, skip = family_matrix(df, "D", "MAR", 0.3, "cat_Macro-F1")
    check(skip == "metric_undefined_for_family",
          "metric absent for all -> undefined skip")
    try:
        family_matrix(df[~((df.method == "KNN") & (df.seed == 5))],
                      "D", "MAR", 0.3, "cont_NRMSE")
        check(False, "partial missing must refuse")
    except ValueError as e:
        check("integrity refusal" in str(e), "partial missing -> refusal")

    # TOST margin formula: crafted SDs -> known delta
    m2 = mat.copy()
    for j, m in enumerate(METHODS):
        m2[m] = [0, 1, 2, 3, 4] if j == 0 else np.arange(5) * (j + 1) * 0.1
    # seed-SDs: col0 sd = 1.5811..., col j: 0.1*(j+1)*1.5811
    sds = m2.std(axis=0, ddof=1).to_numpy()
    check(abs(tost_margin(m2) - 0.5 * float(np.median(sds))) < 1e-12,
          "TOST margin = 0.5 x median seed-SD, exactly")

    # analyze_family: identical columns -> all-zero diffs -> p=1, rb=0,
    # TOST equivalent at any positive margin... margin would be 0.5*sd
    same = pd.DataFrame({m: [0.1, 0.2, 0.3, 0.4, 0.5] for m in METHODS},
                        index=SEEDS)
    fam = analyze_family(same, False)
    check(all(p["wilcoxon_p"] == 1.0 and p["rank_biserial"] == 0.0
              for p in fam["sni_vs_baselines"]),
          "identical methods: p=1, rank-biserial=0 across all pairs")
    # clear separation: SNI worse by a constant >> margin -> not equivalent,
    # Holm-corrected p equals raw p * rank ordering sanity
    sep = same.copy()
    sep["SNI"] = sep["SNI"] + 10.0
    fam2 = analyze_family(sep, False)
    check(all(p["tost"]["equivalent"] is False
              for p in fam2["sni_vs_baselines"]),
          "constant large offset: TOST rejects equivalence for every pair")
    check(all(p["median_delta"] == 10.0 for p in fam2["sni_vs_baselines"]),
          "median delta = the injected +10 exactly")
    # equivalence branch: tiny jitter within margin -> equivalent
    eq = same.copy()
    jit = same.copy()
    jit["SNI"] = jit["SNI"] + 1e-6
    # margin from data ~ 0.5*sd(0.1583)=0.079 >> 1e-6
    fam3 = analyze_family(jit, False)
    check(all(p["tost"]["equivalent"] in (True, None)
              for p in fam3["sni_vs_baselines"]),
          "1e-6 offset inside noise margin: TOST equivalent (or degenerate-recorded)")

    # Holm step-down, exact hand computation: raw [0.01, 0.04, 0.03]
    # sorted: 0.01x3=0.03; 0.03x2=0.06 (monotone); 0.04x1=0.04 -> max=0.06.
    # In input order: [0.03, 0.06, 0.06].
    from stats.posthoc import holm_bonferroni
    hp = holm_bonferroni([0.01, 0.04, 0.03])
    check([round(v, 10) for v in hp] == [0.03, 0.06, 0.06],
          "holm corrected = [0.03, 0.06, 0.06], hand-computed, input order")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "selftest"])
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    return stage_run()


if __name__ == "__main__":
    raise SystemExit(main())
