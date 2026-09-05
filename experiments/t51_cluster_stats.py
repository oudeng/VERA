"""T5.1 -- cluster-robust re-inference, per docs/T51_statistical_analysis_rules.md.

M1: exact seed-block sign-flip (2^n_seeds enumeration, statistic = median
of all paired differences; two-sided p with the identity flip counted).
M2: hierarchical bootstrap CI (seeds then targets/regimes, 10k, seeded).
Leakage: exact McNemar per condition on the 6 paired detections.
Old-caliber values are carried alongside for the before/after table.

    env PYTHONHASHSEED=2025 python experiments/t51_cluster_stats.py --stage selftest
    env PYTHONHASHSEED=2025 python experiments/t51_cluster_stats.py --stage run
"""
from __future__ import annotations

import argparse
import itertools
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

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
T42 = CODE_ROOT / "results" / "T4_leakage"
NP = CODE_ROOT / "results" / "T4_noprior"
OUT = CODE_ROOT / "results" / "T5_stats"

BOOT_SEED = 20260828
N_BOOT = 10_000


# ---------------- pure machinery (exhaustively self-tested) ---------------- #
def sign_flip_exact(diffs_by_block: dict) -> dict:
    """M1 (corrigendum b321d27): enumerate all 2^B joint sign flips;
    TEST statistic = mean of the block (seed) medians -- robust within a
    block, linear across blocks, so identity/mirror are the unique
    extremes for coherent data and the disclosed floor is exact. The
    pooled median is carried as the EFFECT SIZE, not the test statistic."""
    blocks = [np.asarray(v, float) for v in diffs_by_block.values()]
    med_blocks = np.array([float(np.median(b)) for b in blocks])
    obs = float(np.mean(med_blocks))
    B = len(blocks)
    stats = []
    for signs in itertools.product((1.0, -1.0), repeat=B):
        stats.append(float(np.mean(np.asarray(signs) * med_blocks)))
    stats = np.asarray(stats)
    p_two = float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    p_neg = float(np.mean(stats <= obs + 1e-12))
    return {"observed_stat_mean_of_block_medians": obs,
            "effect_pooled_median":
                float(np.median(np.concatenate(blocks))),
            "n_blocks": B, "n_flips": len(stats),
            "p_two_sided": p_two, "p_one_sided_neg": p_neg,
            "floor_two_sided": 2.0 / len(stats)}


def hier_boot_ci(diffs_by_block: dict, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> list:
    """M2: resample blocks with replacement, then units within each drawn
    block; percentile 95% CI of the pooled median."""
    rng = np.random.default_rng(seed)
    blocks = [np.asarray(v, float) for v in diffs_by_block.values()]
    B = len(blocks)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, B, B)
        pooled = np.concatenate([
            blocks[j][rng.integers(0, len(blocks[j]), len(blocks[j]))]
            for j in pick])
        meds[i] = np.median(pooled)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def seed_boot_ci_T(diffs_by_block: dict, n_boot: int = N_BOOT,
                   seed: int = 20260831) -> list:
    """P5R-F SS1.3 estimand unification: 95% CI for the PRIMARY statistic
    T = mean of seed-level medians, bootstrapping SEEDS ONLY (targets /
    regimes are the fixed reference set; the inference scope is training
    randomness). Equivalent to a percentile bootstrap of the mean of the
    block-median vector."""
    rng = np.random.default_rng(seed)
    meds = np.array([float(np.median(np.asarray(v, float)))
                     for v in diffs_by_block.values()])
    B = len(meds)
    stats = np.mean(meds[rng.integers(0, B, (n_boot, B))], axis=1)
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def mcnemar_exact(a_flags, b_flags) -> dict:
    """Exact two-sided McNemar on paired binary outcomes."""
    from scipy.stats import binomtest
    a = np.asarray(a_flags, bool)
    b = np.asarray(b_flags, bool)
    assert len(a) == len(b)
    bb = int(np.sum(a & ~b))
    cc = int(np.sum(~a & b))
    n = bb + cc
    p = 1.0 if n == 0 else float(binomtest(min(bb, cc), n, 0.5,
                                           alternative="two-sided").pvalue)
    return {"n_pairs": int(len(a)), "a_only": bb, "b_only": cc,
            "p_two_sided": p}


def holm(pvals: list) -> list:
    from stats.posthoc import holm_bonferroni
    return [float(x) for x in holm_bonferroni(pvals)]


def branch(p_holm: float) -> str:
    return "S" if p_holm < 0.05 else "N"


def aggregates(diffs_by_block: dict) -> dict:
    """Both aggregation levels, descriptive."""
    seed_medians = {str(k): float(np.median(v))
                    for k, v in diffs_by_block.items()}
    # unit-level: median across blocks per position needs a rectangular
    # layout; callers pass dicts of equal-length arrays keyed by block.
    arrs = np.vstack([np.asarray(v, float) for v in diffs_by_block.values()])
    unit_medians = np.median(arrs, axis=0)
    return {"seed_medians": seed_medians,
            "n_units": int(arrs.shape[1]),
            "unit_median_sign_count":
                {"neg": int(np.sum(unit_medians < 0)),
                 "pos": int(np.sum(unit_medians > 0)),
                 "zero": int(np.sum(unit_medians == 0))}}


# ---------------- family assembly ------------------------------------------ #
SEEDS = [1, 2, 3, 5, 8]
# T51 addendum (commit 9114ab5, fixed before any expanded-seed artifact):
EXP_SEEDS = [13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


def fam_faithfulness(ds: str) -> dict:
    cells = pd.read_csv(FAITH / "faithfulness_cells.csv")
    g = cells[(cells.dataset == ds) & (cells.scope == "full")
              & cells.method.isin(["SNI-D", "P"])]
    piv = g.pivot_table(index=["seed", "target"], columns="method",
                        values="rho")
    d = (piv["SNI-D"] - piv["P"]).unstack("target")
    assert d.shape == (5, len(d.columns)) and not d.isna().any().any()
    return {s: d.loc[s].to_numpy() for s in SEEDS}


def fam_noprior_faith() -> dict:
    from t43_verdict import _row_rhos
    feats = None
    P = None
    out = {}
    for s in SEEDS:
        A = pd.read_csv(FAITH / f"A_NP_eICU_seed{s}_cpu_t2.csv", index_col=0)
        D = pd.read_csv(FAITH / f"D_retrained_NP_eICU_seed{s}_cpu_t2.csv",
                        index_col=0)
        if P is None:
            feats = list(D.index)
            P = pd.read_csv(PRIOR / "P_eICU_seed1_cpu_t2.csv", index_col=0
                            ).reindex(index=feats, columns=feats)
        rd, rp = _row_rhos(D, A), _row_rhos(P, A)
        out[s] = np.array([rd[f] - rp[f] for f in sorted(rd)])
    return out


def fam_recovery(noprior: bool) -> dict:
    from pilot_r21 import load_cell, measured_rows, score
    from t43_noprior_synth import PILOT_METHODS, PILOT
    REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
    SSEEDS = [2025, 2026, 2027, 2028, 2029]
    by_seed = {s: [] for s in SSEEDS}
    for regime in REGIMES:
        for s in SSEEDS:
            complete, *_rest = load_cell(regime, s)
            G = _rest[2]
            cols = list(complete.columns)
            mats = {m: pd.read_csv(PILOT / f"D_{regime}_s{s}_{m}.csv",
                                   index_col=0).reindex(
                index=cols, columns=cols).fillna(0.0) for m in PILOT_METHODS}
            common = np.ones(len(cols), dtype=bool)
            for M in mats.values():
                common &= measured_rows(M)
            if noprior:
                D = pd.read_csv(NP / f"D_NP_{regime}_s{s}.csv", index_col=0
                                ).reindex(index=cols, columns=cols).fillna(0.0)
                common = common & measured_rows(D)
            else:
                D = mats["SNI-D"]
            Pm = pd.read_csv(PRIOR / f"P_synth_{regime}_s{s}.csv", index_col=0
                             ).reindex(index=cols, columns=cols).fillna(0.0)
            by_seed[s].append(float(score(D, G, keep=common)["auroc"])
                              - float(score(Pm, G, keep=common)["auroc"]))
    return {s: np.array(v) for s, v in by_seed.items()}


def fam_leakage() -> dict:
    det = pd.read_csv(T42 / "t42_detection.csv")
    det = det[det.kind == "inj"]
    out = {}
    for cond, g in det.groupby("condition"):
        piv = g.pivot_table(index=["dataset", "host_seed"], columns="object",
                           values="detected")
        out[cond] = mcnemar_exact(piv["SNI-D"].to_numpy(),
                                  piv["P"].to_numpy())
    return out


def fam_faith_expanded(ds: str, variant: str = "SNI",
                       seeds: list | None = None) -> dict:
    """Matrix-direct faithfulness blocks over an arbitrary seed list (the
    5-seed cells CSV stays the frozen pre-expansion artifact; MF readouts
    were not expanded). Rho convention = t43_verdict._row_rhos, the same
    full-scope Spearman the shipped NoPrior family used; for the five
    original SNI seeds the recomputed Delta rho is asserted cell-by-cell
    against faithfulness_cells.csv (same-ruler recompute-assertion,
    tolerance = the CSV's 4-dp rounding on each of the two rhos)."""
    from t43_verdict import _row_rhos
    pre = "" if variant == "SNI" else "NP_"
    cells = pd.read_csv(FAITH / "faithfulness_cells.csv")
    P = None
    out = {}
    for s in (seeds if seeds is not None else SEEDS + EXP_SEEDS):
        tag = f"{pre}{ds}_seed{s}_cpu_t2"
        A = pd.read_csv(FAITH / f"A_{tag}.csv", index_col=0)
        D = pd.read_csv(FAITH / f"D_retrained_{tag}.csv", index_col=0)
        if P is None:
            feats = list(D.index)
            P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                            ).reindex(index=feats, columns=feats)
        rd, rp = _row_rhos(D, A), _row_rhos(P, A)
        targets = sorted(rd)
        out[s] = np.array([rd[f] - rp[f] for f in targets])
        if variant == "SNI" and s in SEEDS:
            g = cells[(cells.dataset == ds) & (cells.scope == "full")
                      & (cells.seed == s)
                      & cells.method.isin(["SNI-D", "P"])]
            piv = g.pivot_table(index="target", columns="method",
                                values="rho")
            ref = (piv["SNI-D"] - piv["P"]).reindex(targets).to_numpy()
            dev = float(np.max(np.abs(out[s] - ref)))
            assert dev <= 1.05e-4, (f"recompute-assertion failed "
                                    f"{ds} seed {s}: max dev {dev:.2e} vs "
                                    f"faithfulness_cells.csv")
    return out


# ---------------- stages ---------------------------------------------------- #
def stage_run() -> int:
    old = json.loads((FAITH / "faithfulness_summary.json").read_text())
    vj = json.loads((NP / "t43_verdict.json").read_text())
    fams = {}
    for ds in ("MIMIC", "eICU"):
        blocks = fam_faithfulness(ds)
        fams[f"faithfulness|{ds}"] = {
            "m1": sign_flip_exact(blocks), "m2_ci95": hier_boot_ci(blocks),
            "aggregates": aggregates(blocks),
            "old_wilcoxon_p_n60": old[ds]["paired_D_minus_P"]["wilcoxon_p"]}
    blocks = fam_noprior_faith()
    fams["noprior_faithfulness|eICU"] = {
        "m1": sign_flip_exact(blocks), "m2_ci95": hier_boot_ci(blocks),
        "aggregates": aggregates(blocks),
        "old_wilcoxon_p_n60": vj["faithfulness_axis"]["wilcoxon_p"]}
    for noprior, name, oldp in (
            (False, "recovery|pilot", None),
            (True, "noprior_recovery|pilot",
             vj["recovery_axis"]["wilcoxon_p"])):
        blocks = fam_recovery(noprior)
        fams[name] = {"m1": sign_flip_exact(blocks),
                      "m2_ci95": hier_boot_ci(blocks),
                      "aggregates": aggregates(blocks),
                      "old_wilcoxon_p_n60": oldp}
    # Holm within the primary family (the two faithfulness tables)
    prim = ["faithfulness|MIMIC", "faithfulness|eICU"]
    hp = holm([fams[k]["m1"]["p_two_sided"] for k in prim])
    for k, p in zip(prim, hp):
        fams[k]["p_holm_primary_family"] = p
        fams[k]["branch"] = branch(p)
    for k in fams:
        if "branch" not in fams[k]:
            fams[k]["branch"] = branch(fams[k]["m1"]["p_two_sided"])
    leak = fam_leakage()
    out = {"rule_doc": "docs/T51_statistical_analysis_rules.md",
           "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
           "families": fams, "leakage_mcnemar_by_condition": leak}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t51_cluster_stats.json").write_text(json.dumps(out, indent=1))
    for k, f in fams.items():
        print(f"{k:28s} stat={f['m1']['observed_stat_mean_of_block_medians']:+.4f} eff={f['m1']['effect_pooled_median']:+.4f} "
              f"p2={f['m1']['p_two_sided']:.4f} "
              f"(floor {f['m1']['floor_two_sided']:.4f}) "
              f"CI{f['m2_ci95']} old_p={f.get('old_wilcoxon_p_n60')} "
              f"branch={f['branch']}")
    for c, m in sorted(leak.items()):
        print(f"leak {c:12s} D-only={m['a_only']} TAP-only={m['b_only']} "
              f"p={m['p_two_sided']:.4f}")
    print(f"[ok] wrote {OUT / 't51_cluster_stats.json'}")
    return 0


def stage_run_redundancy() -> int:
    """B3 fifth family (T51 addendum 9114ab5, pre-written before
    computation): per threshold tau, paired Delta rho (D - TAP) over the
    surviving (target, seed) grid of the excl-redundant scope; seed is the
    block; same M1 + M2 machinery per tau; reported as a sensitivity curve
    with NO per-tau branch or claim of its own. 15 seeds (the family
    inherits the faithfulness expansion). Same-ruler recompute-assertion:
    at the five original seeds the per-tau pooled median must match the
    shipped r1_threshold_sensitivity.csv within its 4-dp rounding."""
    from scipy.stats import spearmanr
    ref = pd.read_csv(FAITH / "r1_threshold_sensitivity.csv")
    out = {}
    for ds in ("MIMIC", "eICU"):
        C = pd.read_csv(FAITH / f"C_pooled_corr_{ds}.csv", index_col=0)
        feats = list(C.index)
        mats = {}
        for s in SEEDS + EXP_SEEDS:
            tag = f"{ds}_seed{s}_cpu_t2"
            mats[s] = (pd.read_csv(FAITH / f"A_{tag}.csv", index_col=0),
                       pd.read_csv(FAITH / f"D_retrained_{tag}.csv",
                                   index_col=0))
        P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                        ).reindex(index=feats, columns=feats)
        for tau in (0.5, 0.6, 0.7, 0.8):
            R = {f for f in feats if float(C.loc[f].drop(f).max()) > tau}
            n_keep = len(feats) - len(R)
            key = f"{ds}|tau={tau}"
            if not (R and n_keep - 1 >= 5):
                out[key] = {"applicable": False, "R_size": len(R),
                            "n_keep": n_keep}
                continue

            def _deltas(seed_list):
                blocks = {}
                for s in seed_list:
                    A, D = mats[s]
                    row = []
                    for f in A.index:
                        kept = [c for c in feats
                                if c != f and c not in R]
                        if len(kept) < 5:
                            continue
                        Arow = A.loc[f, kept].to_numpy(float)
                        rd = float(spearmanr(D.loc[f, kept].to_numpy(float),
                                             Arow).statistic)
                        rp = float(spearmanr(P.loc[f, kept].to_numpy(float),
                                             Arow).statistic)
                        row.append(rd - rp)
                    blocks[s] = np.array(row)
                return blocks

            blocks5 = _deltas(SEEDS)
            med5 = float(np.median(np.concatenate(list(blocks5.values()))))
            stored = ref[(ref.dataset == ds) & (ref.tau == tau)]
            assert abs(med5 - float(stored.median_delta.iloc[0])) <= 1.05e-4, \
                (f"recompute-assertion failed {key}: {med5:.6f} vs stored "
                 f"{float(stored.median_delta.iloc[0])}")
            blocks = _deltas(SEEDS + EXP_SEEDS)
            out[key] = {"applicable": True, "R_size": len(R),
                        "n_keep": n_keep, "n_seeds": len(blocks),
                        "m1": sign_flip_exact(blocks),
                        "m2_ci95": hier_boot_ci(blocks),
                        "aggregates": aggregates(blocks),
                        "old_wilcoxon_p_n60_scope":
                            float(stored.wilcoxon_p.iloc[0])}
    res = {"rule_doc": "docs/T51_statistical_analysis_rules.md "
                       "(addendum 9114ab5, fifth family)",
           "note": "sensitivity curve; no per-tau branch or claim "
                   "(inherits the faithfulness family's reading)",
           "boot_seed": BOOT_SEED, "n_boot": N_BOOT, "cells": out}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t51_redundancy_sensitivity.json").write_text(
        json.dumps(res, indent=1))
    for k, c in out.items():
        if not c.get("applicable"):
            print(f"{k:16s} inapplicable (R={c['R_size']}, keep={c['n_keep']})")
        else:
            print(f"{k:16s} eff={c['m1']['effect_pooled_median']:+.4f} "
                  f"p2={c['m1']['p_two_sided']:.6f} "
                  f"CI[{c['m2_ci95'][0]:+.3f},{c['m2_ci95'][1]:+.3f}] "
                  f"seeds_neg={sum(1 for v in c['aggregates']['seed_medians'].values() if v < 0)}"
                  f"/{c['n_seeds']}")
    print(f"[ok] wrote {OUT / 't51_redundancy_sensitivity.json'}")
    return 0



def stage_run_expanded() -> int:
    """15-seed re-readout per T51 addendum (commit 9114ab5): same units,
    statistic, machinery and branch mapping, enumeration 2^15 (floor
    2/32768); plus the NEW 5-seed MIMIC NoPrior family (P5R SS8). Refuses
    while any expected matrix is missing. Output is a separate file: the
    5-seed t51_cluster_stats.json stays the frozen pre-expansion record,
    and the branch re-selection reads from here."""
    need = []
    for ds in ("MIMIC", "eICU"):
        for s in SEEDS + EXP_SEEDS:
            need.append(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv")
    for s in SEEDS + EXP_SEEDS:
        need.append(FAITH / f"A_NP_eICU_seed{s}_cpu_t2.csv")
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print(f"REFUSING TO RUN: {len(missing)} of {len(need)} matrices "
              f"missing (first: {missing[:3]})", file=sys.stderr)
        return 2
    old5 = json.loads((OUT / "t51_cluster_stats.json").read_text()
                      )["families"]
    fams = {}
    for ds in ("MIMIC", "eICU"):
        blocks = fam_faith_expanded(ds, "SNI")
        fams[f"faithfulness|{ds}"] = {
            "n_seeds": len(blocks),
            "m1": sign_flip_exact(blocks), "m2_ci95": hier_boot_ci(blocks),
            "aggregates": aggregates(blocks),
            "p_5seed_m1_two_sided":
                old5[f"faithfulness|{ds}"]["m1"]["p_two_sided"],
            "old_wilcoxon_p_n60":
                old5[f"faithfulness|{ds}"]["old_wilcoxon_p_n60"]}
    blocks = fam_faith_expanded("eICU", "NoPrior")
    fams["noprior_faithfulness|eICU"] = {
        "n_seeds": len(blocks),
        "m1": sign_flip_exact(blocks), "m2_ci95": hier_boot_ci(blocks),
        "aggregates": aggregates(blocks),
        "p_5seed_m1_two_sided":
            old5["noprior_faithfulness|eICU"]["m1"]["p_two_sided"]}
    # Chat ruling 2026-08-29 (addendum c2e82e2): the NEW MIMIC NoPrior
    # family runs at the full 15 seeds; its readout is withheld until all
    # 15 matrices exist -- no intermediate 5-seed look.
    np_mimic = [FAITH / f"A_NP_MIMIC_seed{s}_cpu_t2.csv"
                for s in SEEDS + EXP_SEEDS]
    have = sum(p.exists() for p in np_mimic)
    if have == len(np_mimic):
        blocks = fam_faith_expanded("MIMIC", "NoPrior")
        fams["noprior_faithfulness|MIMIC"] = {
            "n_seeds": len(blocks),
            "m1": sign_flip_exact(blocks), "m2_ci95": hier_boot_ci(blocks),
            "aggregates": aggregates(blocks)}
    else:
        print(f"[pending] noprior_faithfulness|MIMIC withheld: {have}/"
              f"{len(np_mimic)} matrices (readout only at the full 15, "
              f"addendum c2e82e2)")
    prim = ["faithfulness|MIMIC", "faithfulness|eICU"]
    hp = holm([fams[k]["m1"]["p_two_sided"] for k in prim])
    for k, p in zip(prim, hp):
        fams[k]["p_holm_primary_family"] = p
        fams[k]["branch"] = branch(p)
    for k in fams:
        if "branch" not in fams[k]:
            fams[k]["branch"] = branch(fams[k]["m1"]["p_two_sided"])
    out = {"rule_doc": "docs/T51_statistical_analysis_rules.md "
                       "(addendum 9114ab5)",
           "boot_seed": BOOT_SEED, "n_boot": N_BOOT, "families": fams}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t51_expanded_15seed.json").write_text(json.dumps(out, indent=1))
    for k, f in fams.items():
        print(f"{k:34s} n={f['n_seeds']:2d} "
              f"stat={f['m1']['observed_stat_mean_of_block_medians']:+.4f} "
              f"eff={f['m1']['effect_pooled_median']:+.4f} "
              f"p2={f['m1']['p_two_sided']:.6f} "
              f"(floor {f['m1']['floor_two_sided']:.6f}) "
              f"CI[{f['m2_ci95'][0]:+.3f},{f['m2_ci95'][1]:+.3f}] "
              f"p5={f.get('p_5seed_m1_two_sided')} branch={f['branch']}")
    print(f"[ok] wrote {OUT / 't51_expanded_15seed.json'}")
    return 0


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    # M1 hand-checks
    r = sign_flip_exact({1: [-1.0, -1.0], 2: [-1.0, -1.0]})
    check(r["p_two_sided"] == 0.5 and r["n_flips"] == 4,
          "2 blocks, constant -1: flips (+,-) give median 0 -> p2 = 2/4")
    # Constant blocks tie on |median| for every flip -> p = 1: the exact
    # test is conservative under ties, never anti-conservative.
    rc = sign_flip_exact({s: [-1.0] * 12 for s in range(5)})
    check(abs(rc["p_two_sided"] - 0.0625) < 1e-12,
          "constant blocks: mean-of-medians ties only the mirror -> floor 2/32")
    # Distinct-valued, coherently negative blocks: only the mirror flip
    # ties the identity in |median| -> the floor 2/32 is attained exactly.
    r5 = sign_flip_exact({s: [-(1 + 0.01 * s + 0.001 * j)
                              for j in range(12)] for s in range(5)})
    check(abs(r5["p_two_sided"] - 0.0625) < 1e-12
          and r5["floor_two_sided"] == 0.0625,
          "distinct all-negative blocks: floor p2 = 2/32 attained exactly")
    mixed = sign_flip_exact({1: [-1, -1], 2: [-1, -1], 3: [1, 1],
                             4: [-1, -1], 5: [-1, -1]})
    check(mixed["p_two_sided"] > 0.0625,
          "one opposing block raises p above the floor")

    # T-caliber seed-only bootstrap (P5R-F SS1.3)
    ci_t = seed_boot_ci_T({s: [0.3, 0.3] for s in range(5)}, n_boot=200)
    check(ci_t == [0.3, 0.3], "seed-boot T on constant medians: point CI")
    ci_t2 = seed_boot_ci_T({1: [0.0], 2: [1.0]}, n_boot=4000)
    check(0.0 <= ci_t2[0] <= 0.5 <= ci_t2[1] <= 1.0
          and ci_t2[0] < ci_t2[1],
          "seed-boot T two blocks: interval brackets the mean of medians")

    # M2: degenerate constants -> point interval
    ci = hier_boot_ci({s: [0.3, 0.3, 0.3] for s in range(5)}, n_boot=200)
    check(ci == [0.3, 0.3], "hier bootstrap on constants: point CI")

    # McNemar exact hand-values
    m = mcnemar_exact([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
    check(abs(m["p_two_sided"] - 2 * 0.5 ** 6) < 1e-12 and m["a_only"] == 6,
          "McNemar 6-0 discordant: p = 2*(1/2)^6 = 0.03125")
    m2 = mcnemar_exact([1, 0, 1, 0], [0, 1, 0, 1])
    check(m2["p_two_sided"] == 1.0 and m2["a_only"] == m2["b_only"] == 2,
          "balanced discordance: p = 1")
    m3 = mcnemar_exact([1, 1], [1, 1])
    check(m3["p_two_sided"] == 1.0, "no discordant pairs: p = 1 (defined)")

    # Holm + branch
    check(holm([0.03, 0.0625]) == [0.06, 0.0625],
          "holm over the primary pair, hand-computed")
    check(branch(0.0625) == "N" and branch(0.031) == "S",
          "branch selection strict at 0.05")

    # aggregates shape
    ag = aggregates({1: [-1, -2], 2: [-3, 4]})
    check(ag["n_units"] == 2
          and ag["unit_median_sign_count"] == {"neg": 1, "pos": 1, "zero": 0},
          "aggregates: unit medians (-2, +1) -> one negative, one positive")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["run", "run-expanded", "run-redundancy", "selftest"])
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "run-expanded":
        return stage_run_expanded()
    if a.stage == "run-redundancy":
        return stage_run_redundancy()
    return stage_run()


if __name__ == "__main__":
    raise SystemExit(main())
