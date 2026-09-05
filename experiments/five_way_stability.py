"""T3.1 -- five-way stability on the real tables (the resumed T2g.2, plus P).

The question T2f.1 could not answer alone: SNI-D is stable across seeds on the
real tables (rho 0.947 / 0.891) -- but stable *compared to what*? This puts
every audit object in one coordinate system: SNI-D, the model-free prior P
(bitwise seed-invariant, so its stability is 1.0 by construction -- the P3
ruling requires that column to be given by us rather than asked for), and the
three post-hoc readouts of MissForest, under exactly T2f.1's conditions
(MIMIC / eICU, MAR@30 %, seeds 1,2,3,5,8, CPU, threads pinned).

Three measurements per object (T2g.2's spec):
  * cross-seed stability -- pairwise Spearman over the off-diagonal, 10 pairs,
    plus top-3/top-5 Jaccard, exactly as `d_stability.pairwise` computes them;
  * agreement with P -- per-seed Spearman against the seed-invariant P;
  * cost -- impute seconds and audit seconds, separately.

Row-coverage caveat handled explicitly: the MissForest family yields a row
only for columns MissForest imputed (the 12 masked ones); SNI-D and P have all
16 rows. Stability is therefore reported twice -- on each object's own rows
(SNI-D/P: 16, published T2f convention) and on the 12 rows every object
measures (`rows12`), which is the apples-to-apples column.

    env PYTHONHASHSEED=2025 python experiments/five_way_stability.py --stage compute
    env PYTHONHASHSEED=2025 python experiments/five_way_stability.py --stage analyze
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

STAB = CODE_ROOT / "results" / "T2f_d_stability"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
OUT = CODE_ROOT / "results" / "T3_five_way"

DATASETS = ["MIMIC", "eICU"]
SEEDS = [1, 2, 3, 5, 8]
MF_METHODS = ["MissForest-importance", "SHAP-on-MissForest",
              "Permutation-on-MissForest"]
ALL_METHODS = ["SNI-D", "P"] + MF_METHODS


def stage_compute() -> None:
    from prior_attribution import load_real_case, compute_P
    from pilot_r21 import run_missforest_family

    for ds in DATASETS:
        missing, mask_df, cat, cont = load_real_case(ds)
        feats = list(missing.columns)

        # P's cost, measured once per dataset (P is seed-invariant; T2g.1
        # asserted bitwise identity across all five seeds and all thread arms).
        pcost = OUT / f"cost_P_{ds}.json"
        if not pcost.exists():
            t0 = time.time()
            compute_P(cat, cont, SEEDS[0], missing, mask_df)
            pcost.write_text(json.dumps(
                {"dataset": ds, "method": "P",
                 "impute_sec": 0.0, "audit_sec": round(time.time() - t0, 2)}))
            print(f"[ok] P cost {ds}: {json.loads(pcost.read_text())['audit_sec']}s",
                  flush=True)

        for s in SEEDS:
            done = all((OUT / f"D_{ds}_seed{s}_{m}.csv").exists()
                       for m in MF_METHODS)
            if done:
                print(f"[cached] {ds} seed {s}")
                continue
            fam, no_model = run_missforest_family(missing, cat, cont, s, feats)
            for name, (M, imp_s, aud_s) in fam.items():
                M.to_csv(OUT / f"D_{ds}_seed{s}_{name}.csv")   # per cell (B79)
                rec = {"dataset": ds, "seed": s, "method": name,
                       "impute_sec": round(imp_s, 2), "audit_sec": round(aud_s, 2),
                       "no_model_cols": no_model}
                (OUT / f"M_{ds}_seed{s}_{name}.json").write_text(json.dumps(rec))
            print(f"[ok] {ds} seed {s}  impute={imp_s:.0f}s  "
                  f"no_model={len(no_model)}", flush=True)


# --------------------------------------------------------------------------- #
def _vec(M: pd.DataFrame, keep_rows: np.ndarray) -> np.ndarray:
    A = M.to_numpy(dtype=float)
    sel = keep_rows[:, None] & ~np.eye(len(A), dtype=bool)
    return A[sel]


def _pairs_rows(mats: dict, keep_rows: np.ndarray, group: str,
                variant: str) -> list:
    """d_stability.pairwise's statistics, restricted to keep_rows."""
    from scipy.stats import spearmanr
    rows = []
    d = None
    for a, b in combinations(sorted(mats), 2):
        A, B = mats[a].to_numpy(float), mats[b].to_numpy(float)
        d = len(A)
        off = ~np.eye(d, dtype=bool)
        sel = off & keep_rows[:, None]
        r = spearmanr(A[sel], B[sel])
        rec = {"group": group, "rows": variant, "a": a, "b": b,
               "spearman": round(float(r.statistic), 4), "p": float(r.pvalue)}
        for k in (3, 5):
            ta = {(i, j) for i in range(d) if keep_rows[i]
                  for j in np.argsort(-A[i] * off[i])[:k]}
            tb = {(i, j) for i in range(d) if keep_rows[i]
                  for j in np.argsort(-B[i] * off[i])[:k]}
            rec[f"top{k}_jaccard"] = round(len(ta & tb) / max(len(ta | tb), 1), 4)
        rows.append(rec)
    return rows


def stage_analyze() -> int:
    from scipy.stats import spearmanr

    pair_rows, agree_rows, cost_rows, summary = [], [], [], {}
    for ds in DATASETS:
        D_sni = {s: pd.read_csv(STAB / f"D_{ds}_seed{s}_cpu_t2.csv", index_col=0)
                 for s in SEEDS}
        feats = list(D_sni[SEEDS[0]].index)
        P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                        ).reindex(index=feats, columns=feats)
        mf = {m: {s: pd.read_csv(OUT / f"D_{ds}_seed{s}_{m}.csv", index_col=0
                                 ).reindex(index=feats, columns=feats)
                  for s in SEEDS} for m in MF_METHODS}

        measured12 = ~mf[MF_METHODS[0]][SEEDS[0]].isna().all(axis=1).to_numpy()
        # every MF readout, every seed, must cover the same rows
        for m in MF_METHODS:
            for s in SEEDS:
                got = ~mf[m][s].isna().all(axis=1).to_numpy()
                assert (got == measured12).all(), f"row coverage varies: {m}/{s}"
        all16 = np.ones(len(feats), dtype=bool)
        n12 = int(measured12.sum())

        # ---- stability ---------------------------------------------------- #
        pair_rows += _pairs_rows(D_sni, all16, f"{ds}|SNI-D", "own16")
        pair_rows += _pairs_rows(D_sni, measured12, f"{ds}|SNI-D", f"rows{n12}")
        for m in MF_METHODS:
            mats = {s: mf[m][s].fillna(0.0) for s in SEEDS}
            pair_rows += _pairs_rows(mats, measured12, f"{ds}|{m}", f"rows{n12}")
        # P: seed-invariant bitwise (T2g.1 assertion) => 1.0 by construction.
        pair_rows.append({"group": f"{ds}|P", "rows": "by_construction",
                          "a": "any", "b": "any", "spearman": 1.0, "p": 0.0,
                          "top3_jaccard": 1.0, "top5_jaccard": 1.0})

        # ---- agreement with P -------------------------------------------- #
        for s in SEEDS:
            for name, M in [("SNI-D", D_sni[s])] + [(m, mf[m][s]) for m in MF_METHODS]:
                keep = all16 if name == "SNI-D" else measured12
                r = spearmanr(_vec(M.fillna(0.0), keep), _vec(P, keep))
                agree_rows.append({"dataset": ds, "seed": s, "method": name,
                                   "rows": "own16" if name == "SNI-D" else f"rows{n12}",
                                   "rho_with_P": round(float(r.statistic), 4)})
        agree_rows.append({"dataset": ds, "seed": "all", "method": "P",
                           "rows": "by_construction", "rho_with_P": 1.0})

        # ---- cost --------------------------------------------------------- #
        for s in SEEDS:
            msni = json.loads((STAB / f"M_{ds}_seed{s}_cpu_t2.json").read_text())
            cost_rows.append({"dataset": ds, "seed": s, "method": "SNI-D",
                              "impute_sec": round(float(msni["wall_sec"]), 1),
                              "audit_sec": 0.0})
            for m in MF_METHODS:
                rec = json.loads((OUT / f"M_{ds}_seed{s}_{m}.json").read_text())
                cost_rows.append({k: rec[k] for k in
                                  ("dataset", "seed", "method",
                                   "impute_sec", "audit_sec")})
        cost_rows.append(json.loads((OUT / f"cost_P_{ds}.json").read_text()))

    pairs = pd.DataFrame(pair_rows)
    agree = pd.DataFrame(agree_rows)
    cost = pd.DataFrame(cost_rows)
    pairs.to_csv(OUT / "fiveway_pairs.csv", index=False)
    agree.to_csv(OUT / "fiveway_agreement_with_P.csv", index=False)
    cost.to_csv(OUT / "fiveway_cost.csv", index=False)

    # cross-check: SNI-D own16 must reproduce T2f's published pairs
    for ds in DATASETS:
        ref = pd.read_csv(STAB / f"pairs_seed_{ds}.csv").spearman.to_numpy()
        got = pairs[(pairs.group == f"{ds}|SNI-D")
                    & (pairs.rows == "own16")].spearman.to_numpy()
        if not np.allclose(np.sort(got), np.sort(ref), atol=5e-4):
            print(f"WARNING: SNI-D own16 pairs mismatch vs pairs_seed_{ds}",
                  file=sys.stderr)

    for ds in DATASETS:
        blk = {}
        for m in ALL_METHODS:
            if m == "P":
                blk[m] = {"stability_mean": 1.0, "stability_min": 1.0,
                          "note": "bitwise seed-invariant (T2g.1 assertion)"}
                continue
            rows_tag = "own16" if m == "SNI-D" else pairs[
                (pairs.group == f"{ds}|{m}")].rows.iloc[0]
            g = pairs[(pairs.group == f"{ds}|{m}") & (pairs.rows == rows_tag)]
            g12 = pairs[(pairs.group == f"{ds}|{m}") & (pairs.rows != "own16")]
            blk[m] = {"stability_mean": round(float(g.spearman.mean()), 4),
                      "stability_min": round(float(g.spearman.min()), 4),
                      "stability_rows12_mean": round(float(g12.spearman.mean()), 4),
                      "top3_jaccard_mean": round(float(g.top3_jaccard.mean()), 4),
                      "rho_with_P_mean": round(float(
                          agree[(agree.dataset == ds) & (agree.method == m)
                                ].rho_with_P.astype(float).mean()), 4)}
        summary[ds] = blk
    (OUT / "fiveway_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["compute", "analyze"])
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "compute":
        stage_compute()
        return 0
    return stage_analyze()


if __name__ == "__main__":
    raise SystemExit(main())
