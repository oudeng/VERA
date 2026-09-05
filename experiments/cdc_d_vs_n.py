"""P2e §5 — does CDC2022's dependency matrix D depend on the row count?

Why it is asked. R2-4 is answered with a rho-vs-d curve. CDC2022 is being
down-sampled to n=1000 while every other table keeps n=392..2849, so "D stability
falls as d grows" could be partly "attention means get noisier with fewer rows".
Using a confounded measurement to answer a question about d would hand the
reviewer the objection.

**What this script can and cannot settle.** It compares D's *structure* across
n=500/1000/1500 at one seed. It does **not** measure whether D's *cross-seed
stability* -- the quantity rho-vs-d actually reports -- depends on n. That would
need several seeds at each n (3 x 3 runs, roughly 17 h) rather than three. So a
stable structure here is reassurance, not proof; an unstable one is decisive
against n=1000. The asymmetry is deliberate and is stated in the report.

Metrics, all reported:
  * Spearman between the off-diagonal entries of each pair of D matrices
  * top-K overlap (K = d, i.e. one edge per target on average)
  * sparsity: Gini of each row, and the share of mass in each row's top-3
  * row-mass concentration drift, since a D that flattens as n shrinks would
    weaken rho for reasons unrelated to d

    env PYTHONHASHSEED=2025 python experiments/cdc_d_vs_n.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "results" / "T2e_cdc_d_vs_n"
DATASET = "CDC2022"
N_ROWS = [500, 1000, 1500]
SEED = 1


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def run_one(n: int) -> tuple:
    import yaml
    from baselines.schema import DataSchema
    from common import determinism
    from sni.imputer import SNIConfig, SNIImputer

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{DATASET}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / DATASET / f"{DATASET}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", DATASET)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]

    complete = complete.iloc[:n].reset_index(drop=True)
    mask_df = mask_df.iloc[:n].reset_index(drop=True)
    missing = complete[feats].mask(mask_df)

    proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                            ).read_text())["protocol"]
    epochs = int(proto["epochs"]["SNI"])
    determinism.apply("deterministic", seed=SEED)
    imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                     continuous_vars=list(schema.continuous_vars),
                     config=SNIConfig(seed=SEED, use_gpu=False))   # CPU per P2e 3.1
    imp.cfg.epochs = epochs
    imp.cfg.early_stopping_patience = epochs + 1
    t0 = time.time()
    imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    D = imp.compute_dependency_matrix()
    return D, round(time.time() - t0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=N_ROWS)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    mats, meta = {}, []
    for n in a.rows:
        p = OUT / f"D_n{n}.csv"
        if p.exists():                       # resume: these runs are hours each
            D = pd.read_csv(p, index_col=0)
            sec = None
            print(f"[cached] n={n}", flush=True)
        else:
            D, sec = run_one(n)
            D.to_csv(p)                      # persist immediately (B79)
            print(f"[ok] n={n:<5} {sec}s", flush=True)
        mats[n] = D
        A = D.to_numpy(float)
        off = ~np.eye(len(A), dtype=bool)
        rows_g = [gini(A[i][off[i]]) for i in range(len(A))]
        top3 = [np.sort(A[i][off[i]])[-3:].sum() for i in range(len(A))]
        meta.append({"n": n, "wall_sec": sec, "d": len(A),
                     "mean_row_gini": round(float(np.nanmean(rows_g)), 4),
                     "mean_top3_share": round(float(np.mean(top3)), 4),
                     "frac_entries_below_1e3": round(
                         float((A[off] < 1e-3).mean()), 4)})

    md = pd.DataFrame(meta)
    md.to_csv(OUT / "sparsity_by_n.csv", index=False)
    print("\nsparsity / concentration by n:")
    print(md.to_string(index=False))

    from scipy.stats import spearmanr
    pairs = []
    for x, y in combinations(sorted(mats), 2):
        A, B = mats[x].to_numpy(float), mats[y].to_numpy(float)
        off = ~np.eye(len(A), dtype=bool)
        rho = float(spearmanr(A[off], B[off]).statistic)
        k = len(A)
        ta = set(map(tuple, np.argwhere(A >= np.sort(A[off])[-k])))
        tb = set(map(tuple, np.argwhere(B >= np.sort(B[off])[-k])))
        pairs.append({"n_a": x, "n_b": y, "spearman_offdiag": round(rho, 4),
                      "topK_overlap": round(len(ta & tb) / max(len(ta | tb), 1), 4)})
    pf = pd.DataFrame(pairs)
    pf.to_csv(OUT / "pairwise_by_n.csv", index=False)
    print("\npairwise agreement between D at different n:")
    print(pf.to_string(index=False))

    worst = pf.spearman_offdiag.min()
    print(f"\nlowest pairwise Spearman across n: {worst:.4f}")
    print("  Reading: this bounds how much of a rho-vs-d effect could be an "
          "n effect instead. It does NOT show that cross-seed rho is "
          "n-independent -- that needs several seeds per n and was not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
