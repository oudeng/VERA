"""P2e §6 — the 4x-budget panel: how each method does at its own convergence point.

The main grid answers "same declared budget, both trained to completion", which is
the fair comparison. T2c.3 showed that is not the whole story: on NHANES at 800
epochs TabCSDI reaches R² +0.370 against SNI's +0.344, i.e. **it overtakes SNI
given 4x the budget**, while SNI is already converged at 200 (0.3484 -> 0.3470 ->
0.3440 across 200/400/800, monotonically down). Publishing the curves without
that point would be selective; publishing only a sentence about it wastes a cheap
measurement.

So: both methods at **800 epochs**, seven tables x MAR@30 % x 5 seeds, early
stopping disabled. Both, not just the baseline -- SNI gets slightly worse at 800
and that belongs in the same table. Same ruler, applied to ourselves too.

Device follows the P2e §3.1 ruling: SNI on CPU, TabCSDI on GPU. SNI's numbers are
device-dependent (B83), so the panel and the main grid must use the same device
for a method or the comparison is contaminated.

CDC2022 is at n=1000 per P2e §5.

    env PYTHONHASHSEED=2025 python experiments/budget_panel.py --method TabCSDI
    env PYTHONHASHSEED=2025 python experiments/budget_panel.py --method SNI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "results" / "T2e_budget_panel"
EPOCHS = 800
SEEDS = [1, 2, 3, 5, 8]
#: CDC2022 is down-sampled per P2e §5; every other table is used whole.
ROWS = {"CDC2022": 1000}


def run_cell(method: str, dataset: str, seed: int, use_gpu: bool) -> dict:
    from baselines.schema import DataSchema
    from baselines.registry import build_baseline_imputer
    from common import determinism
    from evaluation.metrics import evaluate_imputation
    from sni.imputer import SNIConfig, SNIImputer

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{dataset}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / dataset / f"{dataset}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]

    n = ROWS.get(dataset)
    if n is not None:
        complete = complete.iloc[:n].reset_index(drop=True)
        mask_df = mask_df.iloc[:n].reset_index(drop=True)
    missing = complete[feats].mask(mask_df)

    determinism.apply("deterministic", seed=seed)
    if method == "SNI":
        imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                         continuous_vars=list(schema.continuous_vars),
                         config=SNIConfig(seed=seed, use_gpu=use_gpu))
        imp.cfg.epochs = EPOCHS
        imp.cfg.early_stopping_patience = EPOCHS + 1
        t0 = time.time()
        X = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    else:
        imp = build_baseline_imputer(method,
                                     categorical_vars=list(schema.categorical_vars),
                                     continuous_vars=list(schema.continuous_vars),
                                     seed=seed, use_gpu=use_gpu, epochs=EPOCHS)
        impl = getattr(imp, "_impl", imp)
        impl.epochs = EPOCHS
        impl.early_stopping_patience = EPOCHS + 1
        t0 = time.time()
        X = imp.impute(missing, schema)
    elapsed = time.time() - t0

    res = evaluate_imputation(X_imputed=X, X_complete=complete[feats],
                              X_missing=missing,
                              categorical_vars=list(schema.categorical_vars),
                              continuous_vars=list(schema.continuous_vars),
                              mask_df=mask_df)
    s = dict(res.summary) if hasattr(res, "summary") else dict(res)
    s.update(method=method, dataset=dataset, seed=seed, epochs=EPOCHS,
             device="cuda" if use_gpu else "cpu",
             n_rows=int(len(complete)), wall_sec=round(elapsed, 1))
    return s


def main() -> int:
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["SNI", "TabCSDI"])
    ap.add_argument("--datasets", nargs="*", default=list(cfg["datasets"]))
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2

    # P2e §3.1: SNI on CPU, TabCSDI on GPU. Not a flag -- getting this wrong would
    # silently make the panel incomparable with the main grid (B83).
    use_gpu = a.method == "TabCSDI"
    OUT.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for ds in a.datasets:
        for seed in a.seeds:
            try:
                r = run_cell(a.method, ds, seed, use_gpu)
            except Exception as exc:
                import traceback
                failures.append({"run": f"{a.method}/{ds}/s{seed}",
                                 "error": repr(exc)[:300],
                                 "traceback": traceback.format_exc()[-800:]})
                print(f"[FAIL] {a.method}/{ds}/s{seed}: {exc!r}", flush=True)
                continue
            rows.append(r)
            # Written per cell. B79 cost 42 of 49 rows by holding results for a
            # final write, and L2 in docs/engineering_lessons.md says so -- yet
            # this script was written an hour later with the same defect. The
            # lesson only counts if it is applied to the next file.
            pd.DataFrame(rows).to_csv(OUT / f"panel_{a.method}.csv", index=False)
            print(f"[ok] {a.method}/{ds}/s{seed:<2} n={r['n_rows']:<5} "
                  f"R2={r.get('cont_R2', float('nan')):>9.4f} "
                  f"NRMSE={r.get('cont_NRMSE', float('nan')):.5f} "
                  f"{r['wall_sec']:7.1f}s", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(OUT / f"panel_{a.method}.csv", index=False)  # final
    (OUT / f"failures_{a.method}.json").write_text(json.dumps(failures, indent=2))
    print(f"\n{len(rows)} ok, {len(failures)} failed -> "
          f"{OUT}/panel_{a.method}.csv")
    # Failures are listed individually and change the exit code: a panel with
    # silent holes is the R0 baselines_deep failure mode.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
