"""P2b decision 1: is TabCSDI's divergence a property of the method or of its budget?

Finding B70 records that TabCSDI's continuous R^2 collapses on three of the six
R0 datasets -- eICU 9/10 runs negative, Concrete 8/10, NHANES 8/10 with a worst
of -223.99 and a mean of -28.90 -- and that our gate reproduced every one of
those runs bit for bit.

Reporting a diverged model as a competitor is the mirror image of the oracle-leak
problem: instead of giving a baseline an unfair advantage we would be giving it
an unfair handicap, and the review asked for fairness in both directions. So
before the divergence is written up as a property of the method, the cheapest
alternative explanation has to be excluded: that the model is simply undertrained.

R0's budget is `epochs=200, diffusion_steps=50`. (The P2b instruction says 100
epochs; the dataclass default in `project_sni_R0/sni/baselines/registry.py:307`
is 200, and that is what ran, so 2x and 4x are 400 and 800.)

Run on the two worst datasets, three seeds each, at 1x / 2x / 4x:

    env PYTHONHASHSEED=2025 python tests/tabcsdi_budget.py

Verdict rule, decided before seeing the numbers:
  * divergence disappears at a higher budget -> it is a budget problem, and the
    whole grid must be re-run at a uniformly raised budget;
  * divergence persists -> it is a property of the method on these tables, and
    this experiment becomes the ESM evidence for saying so.
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
sys.path.insert(0, str(CODE_ROOT))

from baselines.registry import build_baseline_imputer   # noqa: E402
from baselines.schema import DataSchema                 # noqa: E402
from common import determinism, runconfig               # noqa: E402
from sni.metrics import evaluate_imputation             # noqa: E402

R0_EPOCHS = 200
MULTIPLIERS = [1, 2, 4]
DATASETS = ["NHANES", "Concrete"]
SEEDS = [1, 2, 3]
OUT = CODE_ROOT / "results" / "T2b_tabcsdi_budget"


def _load(dataset: str, mask_root: Path, table_root: Path):
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    blk = cfg["datasets"][dataset]
    complete = pd.read_csv(table_root / f"{dataset}_complete.csv")
    tag = f"{dataset}_MAR_30per"
    mask = np.load(mask_root / dataset / f"{tag}_mask.npy").astype(bool)
    missing = complete.mask(pd.DataFrame(mask, columns=complete.columns))
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    return complete, missing, mask, schema, blk


def run_one(dataset: str, seed: int, mult: int, mask_root: Path,
            table_root: Path) -> dict:
    complete, missing, mask, schema, blk = _load(dataset, mask_root, table_root)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)

    determinism.apply("deterministic", seed=seed)
    imp = build_baseline_imputer(
        "TabCSDI",
        categorical_vars=list(schema.categorical_vars),
        continuous_vars=list(schema.continuous_vars),
        seed=seed, use_gpu=True, epochs=R0_EPOCHS * mult)

    t0 = time.time()
    X_imp = imp.impute(missing[feats], schema)
    elapsed = time.time() - t0

    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    res = evaluate_imputation(
        X_imputed=X_imp, X_complete=complete[feats], X_missing=missing[feats],
        categorical_vars=list(schema.categorical_vars),
        continuous_vars=list(schema.continuous_vars), mask_df=mask_df)
    s = dict(res.summary) if hasattr(res, "summary") else dict(res)
    s.update(dataset=dataset, seed=seed, multiplier=mult,
             epochs=R0_EPOCHS * mult, runtime_sec=elapsed)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--multipliers", type=int, nargs="*", default=MULTIPLIERS)
    ap.add_argument("--masks", default=str(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"))
    ap.add_argument("--tables", default=str(CODE_ROOT / "data" / "derived_shuffled"))
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for ds in a.datasets:
        for mult in a.multipliers:
            for seed in a.seeds:
                tag = f"{ds} x{mult} s{seed}"
                try:
                    r = run_one(ds, seed, mult, Path(a.masks), Path(a.tables))
                except Exception as exc:               # listed, never skipped
                    failures.append({"run": tag, "error": repr(exc)[:300]})
                    print(f"[FAIL] {tag}: {exc!r}", flush=True)
                    continue
                rows.append(r)
                print(f"[OK] {tag:<18} epochs={r['epochs']:<4} "
                      f"R2={r.get('cont_R2', float('nan')):>12.4f} "
                      f"NRMSE={r.get('cont_NRMSE', float('nan')):.6f} "
                      f"{r['runtime_sec']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "budget_runs.csv", index=False)
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2))

    if not df.empty and "cont_R2" in df:
        print("\n" + "=" * 68)
        g = df.groupby(["dataset", "multiplier"])["cont_R2"]
        summary = g.agg(["count", "median", "min", "max"]).reset_index()
        summary["n_negative"] = (df.groupby(["dataset", "multiplier"])["cont_R2"]
                                 .apply(lambda s: int((s < 0).sum())).to_numpy())
        print(summary.to_string(index=False))
        summary.to_csv(OUT / "budget_summary.csv", index=False)

        print("\nVERDICT (rule fixed before the run):")
        for ds in df.dataset.unique():
            sub = summary[summary.dataset == ds].sort_values("multiplier")
            base = sub.iloc[0]
            best = sub.iloc[-1]
            if base.n_negative > 0 and best.n_negative == 0:
                v = "BUDGET PROBLEM -- divergence clears at a higher budget"
            elif best.n_negative > 0:
                v = "METHOD PROPERTY -- still diverges at 4x"
            else:
                v = "no divergence at any budget"
            print(f"  {ds:<10} {v}")
    if failures:
        print(f"\n{len(failures)} run(s) FAILED, listed in failures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
