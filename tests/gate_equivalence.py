"""T2.0(a) equivalence gate: R0's original code vs the code_SNI port, same conditions.

P1 left one confounder in place. R0's per-feature seed is
``cfg.seed + hash(feature) % 10000`` and CPython randomises string hashing per
process, so "R0 seed 1" is not a reproducible object (finding B48). Every P1
number was therefore a distribution comparison, not an identity check.

This gate removes that confounder by fixing ``PYTHONHASHSEED`` **before the
interpreter starts** on both sides, so ``hash(feature)`` is pinned and the two
codebases can be compared directly.

Why two configurations
----------------------
The P2 instruction asks for ``deterministic`` mode on both sides. R0's code has
no such switch: ``impute()`` hardcodes ``if use_gpu: enable_performance_mode()``
(imputer.py:240-242), which turns TF32 on, and R0 is frozen and read-only. So:

``cpu``
    Both sides with ``use_gpu=False``. R0 then never calls
    ``enable_performance_mode`` at all, and CPU arithmetic is deterministic, so a
    faithful port must be **bit-identical**. This is the decisive test.

``gpu``
    Both sides on GPU. The port runs in ``performance`` mode, which reproduces
    R0's effective settings exactly: R0 does
    ``set_global_seed`` (cudnn deterministic=True, benchmark=False) then
    ``enable_performance_mode`` (deterministic=False, benchmark=True, TF32=True);
    the port's ``performance`` mode sets the same end state and leaves
    ``use_deterministic_algorithms`` at its default False. This matches the
    configuration that actually produced the published numbers, at the cost of
    TF32 and non-deterministic attention kernels putting a floor on agreement --
    which is why ``--self-check`` measures that floor.

Usage
-----
    env PYTHONHASHSEED=2025 python code_SNI/tests/gate_equivalence.py \
        --side r0   --device cpu --datasets MIMIC --seeds 1
    env PYTHONHASHSEED=2025 python code_SNI/tests/gate_equivalence.py \
        --side port --device cpu --datasets MIMIC --seeds 1
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_EQ_THREADS", "32")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NT)
# Required for deterministic cuBLAS GEMM; harmless on CPU. Must precede torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(_NT))

CODE_ROOT = Path(__file__).resolve().parent.parent
WS = CODE_ROOT.parent
R0_TREE = WS / "project_sni_R0" / "sni"

# Hyperparameters exactly as in project_sni_R0/sni/data/manifest_sni_v03_main.csv.
R0_HP = dict(
    variant="SNI", hard_prior_lambda=10.0, alpha0=1.0, gamma=0.9,
    max_iters=3, tol=1e-4, use_stat_refine=True, mask_fraction=0.15,
    hidden_dims=(256, 128, 64), emb_dim=128, num_heads=16,
    lr=2e-4, epochs=200, batch_size=128,
    cat_balance_mode="none", cat_lr_mult=1.0,
    lambda_mode="learned", lambda_fixed_value=1.0,
)

DATASETS = {
    "MIMIC": dict(
        complete="data/MIMIC_complete.csv",
        missing="data/MIMIC/MIMIC_MAR_30per.csv",
        categorical=["SpO2", "ALARM"],
        continuous=["RESP", "ABP", "SBP", "DBP", "HR", "PULSE"],
    ),
    "NHANES": dict(
        complete="data/NHANES_complete.csv",
        missing="data/NHANES/NHANES_MAR_30per.csv",
        categorical=["gender_std", "age_band"],
        continuous=["waist_circumference", "systolic_bp", "diastolic_bp",
                    "triglycerides", "hdl_cholesterol", "fasting_glucose",
                    "age", "bmi", "hba1c", "metabolic_score"],
    ),
}

METRICS = ["cont_NRMSE", "cont_RMSE", "cont_MAE", "cont_MB", "cont_R2",
           "cont_Spearman", "cat_Accuracy", "cat_Macro-F1", "cat_Cohen_kappa",
           "lambda_mean", "lambda_std"]


def _import_side(side: str):
    """Import one codebase. The two use different package names, so no clash."""
    if side == "r0":
        sys.path.insert(0, str(R0_TREE))
        from SNI_v0_3.imputer import SNIConfig, SNIImputer      # noqa: E402
        from SNI_v0_3.metrics import (evaluate_imputation,       # noqa: E402
                                      augment_summary_with_imputer_stats)
        from SNI_v0_3.dataio import (load_complete_and_missing,  # noqa: E402
                                     cast_dataframe_to_schema)
    else:
        sys.path.insert(0, str(CODE_ROOT))
        from sni.imputer import SNIConfig, SNIImputer            # noqa: E402
        from sni.metrics import (evaluate_imputation,            # noqa: E402
                                 augment_summary_with_imputer_stats)
        from sni.dataio import (load_complete_and_missing,       # noqa: E402
                                cast_dataframe_to_schema)
    return (SNIConfig, SNIImputer, evaluate_imputation,
            augment_summary_with_imputer_stats, load_complete_and_missing,
            cast_dataframe_to_schema)


def run_one(side: str, dataset: str, seed: int, device: str, outroot: Path,
            rep: int = 0) -> dict:
    (SNIConfig, SNIImputer, evaluate_imputation, augment, load_pair,
     cast_schema) = _import_side(side)

    spec = DATASETS[dataset]
    cat, cont = spec["categorical"], spec["continuous"]

    X_complete, X_missing, schema = load_pair(
        input_complete=str(R0_TREE / spec["complete"]),
        input_missing=str(R0_TREE / spec["missing"]),
        categorical_vars=cat, continuous_vars=cont,
    )
    mask_df = X_missing.isna()

    use_gpu = (device == "gpu")
    kw = dict(R0_HP)
    kw["use_gpu"] = use_gpu
    if side == "port":
        # performance mode reproduces R0's effective torch state exactly; on CPU
        # the flag is inert, so it is set unconditionally for symmetry.
        kw["determinism_mode"] = "performance"
    cfg = SNIConfig(seed=seed, **kw)

    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont, config=cfg)
    t0 = time.time()
    # R0's runner passed X_complete, but imputer.py only reindexes and casts it
    # into a local that is never read again, so passing None is provably a no-op
    # and keeps the two sides on exactly the same code path.
    X_imp = imp.impute(X_missing=X_missing, X_complete=None, mask_df=mask_df)
    runtime = time.time() - t0

    X_imp = cast_schema(X_imp, schema)
    res = evaluate_imputation(X_imputed=X_imp, X_complete=X_complete,
                              X_missing=X_missing, categorical_vars=cat,
                              continuous_vars=cont, mask_df=mask_df)
    summary = augment(dict(res.summary), imp)
    summary.update(dict(side=side, dataset=dataset, seed=seed, device=device,
                        rep=rep, runtime_sec_wall=runtime,
                        hash_probe=hash("RESP") % 10000,
                        pythonhashseed=os.environ.get("PYTHONHASHSEED")))

    tag = f"{dataset}_s{seed}" + (f"_r{rep}" if rep else "")
    outdir = outroot / f"{device}_{side}" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    imp.compute_dependency_matrix().to_csv(outdir / "dependency_matrix.csv")
    # Full float64 precision: the gate's threshold is 1e-6, so the artifact must
    # not be the thing that rounds.
    X_imp.to_csv(outdir / "imputed.csv", index=False, float_format="%.17g")

    print(f"[OK] {side:4s} {device} {dataset} s{seed}"
          + (f" r{rep}" if rep else "")
          + f"  {runtime:6.1f}s  NRMSE={summary['cont_NRMSE']:.12f}"
            f"  Acc={summary['cat_Accuracy']:.12f}  hash={summary['hash_probe']}",
          flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["r0", "port"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--datasets", nargs="+", default=["MIMIC", "NHANES"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--rep", type=int, default=0,
                    help=">0 writes to a separate dir, for self-reproducibility checks")
    ap.add_argument("--outroot", default=str(CODE_ROOT / "results" / "T2.0_gate"))
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts, e.g. `env PYTHONHASHSEED=2025 python ...`. Setting it from "
              "inside the process has no effect -- that is finding B48 itself.",
              file=sys.stderr)
        return 2

    rows = [run_one(a.side, ds, s, a.device, Path(a.outroot), a.rep)
            for ds in a.datasets for s in a.seeds]
    out = Path(a.outroot) / f"summary_{a.device}_{a.side}{'_r%d' % a.rep if a.rep else ''}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[DONE] {len(rows)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
