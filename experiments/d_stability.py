"""T2f.1 / T2f.2 — is D a reproducible object on real tables?

The pilot (T2.5) found SNI-D's cross-seed Spearman at **0.153** on R0's synthetic
set, against R0's published claim of **0.951** on MIMIC and **0.633** on eICU. But
those two numbers have never been compared in the same units: the pilot measured
synthetic tables and the paper measured real ones, and in the pilot the *post-hoc
comparators were also unstable* (median rho about 0.26), which says the synthetic
benchmark is unkind to every method of this kind. So the pilot cannot settle the
paper's claim. This can.

Two measurements, deliberately different in what they vary:

**T2f.1, cross-seed (`--mode seed`).** MIMIC and eICU, MAR@30 %, 5 seeds, 200
epochs, early stopping off, CPU, threads pinned. Pairwise Spearman over the
off-diagonal, C(5,2)=10 pairs, reported exactly as R0's Table 3 does, plus the
top-k reading (k=3, 5) that R0's own discussion suggests for wider tables.

**T2f.2, numerical perturbation (`--mode perturb`).** MIMIC, **seed fixed at 1**,
varying only things that should change nothing: thread count (1, 8, 24) and device
(CUDA). This is the carrier-controlled comparison, and it is what separates two very different
worlds:

  * rho_thread and rho_device high, rho_seed low  -> D varies with the pseudo-mask
    draw. A real property, explainable and disclosable.
  * **rho_thread or rho_device also low -> D is not a function of the data at
    all**, and cannot be published as an audit artifact.

Everything is written per cell (B79) and re-runs resume from cached matrices,
because each cell is a full 200-epoch SNI training.

    env PYTHONHASHSEED=2025 python experiments/d_stability.py --mode seed
    env PYTHONHASHSEED=2025 python experiments/d_stability.py --mode perturb
"""

from __future__ import annotations

import os

# Thread count is a controlled variable (B84) and must be set before torch. In
# --mode perturb it is the independent variable, so it comes from the environment.
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
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "results" / "T2f_d_stability"
SEEDS = [1, 2, 3, 5, 8]
#: R0's published cross-seed D stability, for side-by-side comparison.
R0_CLAIM = {"MIMIC": 0.951, "eICU": 0.633}


def run_cell(dataset: str, seed: int, use_gpu: bool, tag: str) -> tuple:
    """One SNI fit; returns (D, metrics, seconds). Cached by tag."""
    import yaml
    from baselines.schema import DataSchema
    from common import determinism
    from evaluation.metrics import evaluate_imputation
    from sni.imputer import SNIConfig, SNIImputer
    import torch

    dpath = OUT / f"D_{tag}.csv"
    mpath = OUT / f"M_{tag}.json"
    if dpath.exists() and mpath.exists():
        return pd.read_csv(dpath, index_col=0), json.loads(mpath.read_text()), None

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{dataset}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / dataset / f"{dataset}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    missing = complete[feats].mask(mask_df)

    proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                            ).read_text())["protocol"]
    epochs = int(proto["epochs"]["SNI"])
    determinism.apply("deterministic", seed=seed)
    imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                     continuous_vars=list(schema.continuous_vars),
                     config=SNIConfig(seed=seed, use_gpu=use_gpu))
    imp.cfg.epochs = epochs
    imp.cfg.early_stopping_patience = epochs + 1

    t0 = time.time()
    X = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    sec = time.time() - t0
    D = imp.compute_dependency_matrix()

    res = evaluate_imputation(X_imputed=X, X_complete=complete[feats],
                              X_missing=missing,
                              categorical_vars=list(schema.categorical_vars),
                              continuous_vars=list(schema.continuous_vars),
                              mask_df=mask_df)
    m = dict(res.summary) if hasattr(res, "summary") else dict(res)
    m.update(dataset=dataset, seed=seed, device="cuda" if use_gpu else "cpu",
             threads_requested=int(_NT), threads_actual=int(torch.get_num_threads()),
             wall_sec=round(sec, 1), tag=tag)
    D.to_csv(dpath)                    # per cell (B79)
    mpath.write_text(json.dumps(m, indent=2, default=str))
    return D, m, sec


def offdiag(M: pd.DataFrame) -> np.ndarray:
    A = M.to_numpy(dtype=float)
    return A[~np.eye(len(A), dtype=bool)]


def pairwise(mats: dict, label: str) -> pd.DataFrame:
    """Spearman over the off-diagonal, and the top-k readings R0's text suggests."""
    from scipy.stats import spearmanr

    rows = []
    for a, b in combinations(sorted(mats), 2):
        A, B = mats[a].to_numpy(float), mats[b].to_numpy(float)
        off = ~np.eye(len(A), dtype=bool)
        r = spearmanr(A[off], B[off])
        rec = {"group": label, "a": a, "b": b,
               "spearman": round(float(r.statistic), 4),
               "p": float(r.pvalue)}
        for k in (3, 5):
            # Top-k per row: the reading R0's discussion recommends for wide tables.
            ta = {(i, j) for i in range(len(A))
                  for j in np.argsort(-A[i] * off[i])[:k]}
            tb = {(i, j) for i in range(len(B))
                  for j in np.argsort(-B[i] * off[i])[:k]}
            rec[f"top{k}_jaccard"] = round(len(ta & tb) / max(len(ta | tb), 1), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seed", "perturb"], required=True)
    ap.add_argument("--datasets", nargs="*", default=["MIMIC", "eICU"])
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--label", default=None,
                    help="perturb mode: the arm's name, e.g. cpu_t8")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    if a.mode == "seed":
        for ds in a.datasets:
            mats, metrics = {}, []
            for s in a.seeds:
                D, m, sec = run_cell(ds, s, a.device == "cuda",
                                     f"{ds}_seed{s}_{a.device}_t{_NT}")
                mats[str(s)] = D
                metrics.append(m)
                print(f"[ok] {ds} seed {s:<2} "
                      f"R2={m.get('cont_R2', float('nan')):>9.4f} "
                      f"{'cached' if sec is None else f'{sec:.0f}s'}", flush=True)
            pf = pairwise(mats, ds)
            pf.to_csv(OUT / f"pairs_seed_{ds}.csv", index=False)
            pd.DataFrame(metrics).to_csv(OUT / f"metrics_seed_{ds}.csv", index=False)
            print(f"\n{ds}: {len(pf)} seed pairs")
            print(pf.to_string(index=False))
            print(f"  mean {pf.spearman.mean():.4f}  min {pf.spearman.min():.4f}  "
                  f"max {pf.spearman.max():.4f}  "
                  f"%pairs p<0.05 {100 * (pf.p < 0.05).mean():.0f}%")
            print(f"  R0 published for {ds}: {R0_CLAIM.get(ds, 'n/a')}\n", flush=True)
    else:
        # One arm per invocation: the thread count must be set before this process
        # imported torch, so the arms cannot share a process.
        label = a.label or f"{a.device}_t{_NT}"
        ds = a.datasets[0]
        D, m, sec = run_cell(ds, a.seeds[0], a.device == "cuda",
                             f"{ds}_perturb_{label}")
        print(f"[ok] {ds} perturb arm {label}: "
              f"R2={m.get('cont_R2', float('nan')):.4f} "
              f"Macro-F1={m.get('cat_Macro-F1', float('nan')):.4f} "
              f"threads_actual={m['threads_actual']} "
              f"{'cached' if sec is None else f'{sec:.0f}s'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
