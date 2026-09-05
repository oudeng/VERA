"""T4.3 recovery axis -- No-Prior SNI's D on the pilot's 15 synthetic cells.

Paper Y's claim is about attention-derived reliance matrices as a practice,
not about one prior-regularized module. This runs the NoPrior variant
(alpha = 0: attention trained with no prior injection) on exactly the pilot's
cells, scores its D against the same ground truth with the pilot's own
score()/measured_rows(), on the same common row set the pilot used
(recomputed from the four stored matrices, intersected with NoPrior's own
coverage). Cached per cell (B79).

    env PYTHONHASHSEED=2025 python experiments/t43_noprior_synth.py [--seeds ...]
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

PILOT = CODE_ROOT / "results" / "T2.5_pilot"
OUT = CODE_ROOT / "results" / "T4_noprior"

REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SYNTH_SEEDS = [2025, 2026, 2027, 2028, 2029]
PILOT_METHODS = ["SNI-D", "MissForest-importance", "SHAP-on-MissForest",
                 "Permutation-on-MissForest"]


def run_cell(regime: str, seed: int) -> dict:
    import yaml
    from common import determinism
    from pilot_r21 import load_cell, measured_rows, row_normalise, score
    from sni.imputer import SNIConfig, SNIImputer

    complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
    cols = list(complete.columns)
    dpath = OUT / f"D_NP_{regime}_s{seed}.csv"

    if dpath.exists():
        D = pd.read_csv(dpath, index_col=0)
        sec = None
    else:
        proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                                ).read_text())["protocol"]
        determinism.apply("deterministic", seed=seed)
        imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                         config=SNIConfig(seed=seed, use_gpu=False))
        imp.cfg.epochs = int(proto["epochs"]["SNI"])
        imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
        imp.cfg.variant = "NoPrior"
        t0 = time.time()
        imp.impute(X_missing=missing[cols], X_complete=None,
                   mask_df=mask[cols])
        sec = time.time() - t0
        D = imp.compute_dependency_matrix().reindex(
            index=cols, columns=cols).fillna(0.0)
        row_normalise(D).to_csv(dpath)          # pilot's stored convention
        D = pd.read_csv(dpath, index_col=0)

    mats = {m: pd.read_csv(PILOT / f"D_{regime}_s{seed}_{m}.csv", index_col=0
                           ).reindex(index=cols, columns=cols).fillna(0.0)
            for m in PILOT_METHODS}
    common = np.ones(len(cols), dtype=bool)
    for M in mats.values():
        common &= measured_rows(M)
    common &= measured_rows(D)

    sc = score(D, G, keep=common)
    sc_sni = score(mats["SNI-D"], G, keep=common)   # same-rows reference
    return {"regime": regime, "seed": seed, "wall_sec": None if sec is None
            else round(sec, 1), "n_rows_scored": int(common.sum()),
            **{f"NP_{k}": v for k, v in sc.items()},
            **{f"SNID_{k}": v for k, v in sc_sni.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--seeds", type=int, nargs="*", default=SYNTH_SEEDS)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in a.regimes:
        for s in a.seeds:
            rec = run_cell(regime, s)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(OUT / "t43_noprior_synth_cells.csv",
                                      index=False)   # per cell (B79)
            print(f"[ok] {regime} s{s} NP_auroc={rec['NP_auroc']:.4f} "
                  f"(SNI-D same rows {rec['SNID_auroc']:.4f}) "
                  f"{'cached' if rec['wall_sec'] is None else str(rec['wall_sec'])+'s'}",
                  flush=True)
    df = pd.DataFrame(rows)
    print("\nmean AUROC by regime:")
    print(df.groupby("regime")[["NP_auroc", "SNID_auroc"]].mean().round(4)
          .to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
