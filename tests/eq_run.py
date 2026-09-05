"""T1.3 equivalence verification: does the ported SNI reproduce R0?

Runs the ported ``code_SNI.sni.SNIImputer`` on R0's own masks with R0's own
hyperparameters, and writes one result directory per (dataset, seed, condition).
The comparison against ``project_sni_R0/results_all/sni_v03_main/`` is done by
``equivalence_report.py``; this script only produces runs.

Conditions
----------
``r0_performance``
    determinism_mode="performance" and hash randomization left ON — this is
    exactly what R0 did. Note that R0's per-feature reseeding uses
    ``hash(feature_name)``, which CPython randomises per process, so this
    condition is not expected to be reproducible even against itself. That is
    finding B48 and measuring it is one of the points of this script.
``deterministic``
    determinism_mode="deterministic" with ``PYTHONHASHSEED`` fixed before the
    interpreter starts, i.e. what code_SNI will do from now on.

Usage
-----
    PYTHONPATH=$PWD \
    python code_SNI/tests/equivalence_run.py --condition deterministic \
           --datasets MIMIC NHANES --seeds 1 2 3 5 8
"""

from __future__ import annotations

import os

# Thread caps must be set BEFORE numpy/torch are imported, which is also where
# R0's batch runner set them (run_manifest_parallel.py:41-46 for the BLAS/OMP
# variables, :173-182 for torch.set_num_threads).
#
# MEASURED, and not what we first assumed. One MIMIC run on an idle machine:
#
#   1 process,  32 threads          269 s   (reproduced twice, identical outputs)
#   2 processes, 8 threads each   > 6 h     (no completion; both at 100% of one core)
#   2 processes, 8 threads each    4249 s   (under additional load)
#  10 processes, 1 thread each    > 70 min each
#
# The dominant factor is how many processes share the single GPU, not the thread
# count: two concurrent processes make each run 13-15x slower even when 30 cores
# sit idle. This is the same effect that killed all 300 baselines_deep runs in R0
# with "CUDA-capable device(s) is/are busy or unavailable".
#
# So: keep total threads at or below the core count, and above all DO NOT run more
# than one of these processes per GPU. Default 32 = run solo.
_NT = os.environ.get("SNI_EQ_THREADS", "32")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NT)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(_NT))
try:
    torch.set_num_interop_threads(int(_NT))
except RuntimeError:
    pass  # already initialized

CODE_ROOT = Path(__file__).resolve().parent.parent
R0 = CODE_ROOT.parent / "project_sni_R0" / "sni"
sys.path.insert(0, str(CODE_ROOT))

from common import masks as maskmod          # noqa: E402
from common import runconfig                  # noqa: E402
from sni.imputer import SNIConfig, SNIImputer  # noqa: E402
from sni.metrics import evaluate_imputation, augment_summary_with_imputer_stats  # noqa: E402
from sni.dataio import load_complete_and_missing, cast_dataframe_to_schema  # noqa: E402

# Hyperparameters exactly as in project_sni_R0/sni/data/manifest_sni_v03_main.csv
# (all 60 rows identical; verified in P1/T1.2).
R0_HP = dict(
    variant="SNI", hard_prior_lambda=10.0, alpha0=1.0, gamma=0.9,
    max_iters=3, tol=1e-4, use_stat_refine=True, mask_fraction=0.15,
    hidden_dims=(256, 128, 64), emb_dim=128, num_heads=16,
    lr=2e-4, epochs=200, batch_size=128, use_gpu=True,
    cat_balance_mode="none", cat_lr_mult=1.0,
    lambda_mode="learned", lambda_fixed_value=1.0,
)

DATASETS = {
    "MIMIC": dict(
        complete="data/MIMIC_complete.csv",
        missing="data/MIMIC/MIMIC_MAR_30per.csv",
        mask="data/MIMIC/MIMIC_MAR_30per_mask.npy",
        categorical=["SpO2", "ALARM"],
        continuous=["RESP", "ABP", "SBP", "DBP", "HR", "PULSE"],
    ),
    "NHANES": dict(
        complete="data/NHANES_complete.csv",
        missing="data/NHANES/NHANES_MAR_30per.csv",
        mask="data/NHANES/NHANES_MAR_30per_mask.npy",
        categorical=["gender_std", "age_band"],
        continuous=["waist_circumference", "systolic_bp", "diastolic_bp",
                    "triglycerides", "hdl_cholesterol", "fasting_glucose",
                    "age", "bmi", "hba1c", "metabolic_score"],
    ),
}


def run_one(dataset: str, seed: int, condition: str, outroot: Path, repeat: int = 0) -> dict:
    spec = DATASETS[dataset]
    cat, cont = spec["categorical"], spec["continuous"]
    allv = cat + cont

    X_complete, X_missing, schema = load_complete_and_missing(
        input_complete=str(R0 / spec["complete"]),
        input_missing=str(R0 / spec["missing"]),
        categorical_vars=cat,
        continuous_vars=cont,
    )

    # E4: the cached .npy is the authority; assert it agrees with the CSV's NaNs.
    # The stored mask covers every column of the generated table, ID included, so
    # it must be loaded with the full column list of the on-disk missing table.
    raw_missing = pd.read_csv(R0 / spec["missing"])
    mask_full, mask_check = maskmod.load_and_verify(
        raw_missing[[c for c in raw_missing.columns]],
        R0 / spec["mask"],
        columns=list(raw_missing.columns),
        strict=True,
    )
    mask_df = mask_full[allv].copy()
    mask_df.index = X_missing.index

    det_mode = "performance" if condition == "r0_performance" else "deterministic"
    cfg = SNIConfig(seed=seed, determinism_mode=det_mode, **R0_HP)

    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont, config=cfg)
    t0 = time.time()
    # X_complete is deliberately not passed. In R0 the runner did pass it, but
    # imputer.py only reindexes and casts it into a local (lines 264-265, 279-280)
    # that is never read again — grep confirms X_complete appears nowhere else in
    # the file. So this is provably a no-op, and it makes explicit that SNI, unlike
    # every R0 baseline, never touches ground truth (P0 finding B6).
    X_imputed = imp.impute(X_missing=X_missing, X_complete=None, mask_df=mask_df)
    runtime = time.time() - t0

    X_imputed = cast_dataframe_to_schema(X_imputed, schema)
    res = evaluate_imputation(
        X_imputed=X_imputed, X_complete=X_complete, X_missing=X_missing,
        categorical_vars=cat, continuous_vars=cont, mask_df=mask_df,
    )
    summary = dict(res.summary)
    summary = augment_summary_with_imputer_stats(summary, imp)

    tag = f"{dataset}_MAR_30per_SNI_s{seed}" + (f"_r{repeat}" if repeat else "")
    outdir = outroot / condition / tag
    outdir.mkdir(parents=True, exist_ok=True)

    summary.update(dict(dataset=dataset, seed=seed, condition=condition,
                        repeat=repeat, runtime_sec_wall=runtime,
                        hash_offset_probe=hash("RESP") % 10000,
                        pythonhashseed=os.environ.get("PYTHONHASHSEED")))
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    imp.compute_dependency_matrix().to_csv(outdir / "dependency_matrix.csv")
    imp.get_lambda_per_head_df().to_csv(outdir / "lambda_per_head.csv", index=False)
    X_imputed.to_csv(outdir / "imputed.csv", index=False)

    runconfig.write(outdir, runconfig.build(
        exp_id=tag, method="SNI",
        params={**R0_HP, "determinism_mode": det_mode},
        inputs={"complete": str(R0 / spec["complete"]),
                "missing": str(R0 / spec["missing"]),
                "mask": str(R0 / spec["mask"]),
                "mask_check": mask_check.__dict__},
        seeds={"model": seed, "mask_generator": 2025},
        determinism=imp.determinism_state_,
        extra={"purpose": "P1/T1.3 equivalence verification",
               "r0_reference": f"project_sni_R0/results_all/sni_v03_main/V03_MAIN_{dataset}_MAR_30per_SNI_s{seed}"},
    ))
    print(f"[OK] {condition}/{tag}  {runtime:.1f}s  NRMSE={summary.get('cont_NRMSE'):.6f} "
          f"Acc={summary.get('cat_Accuracy'):.6f} lam={summary.get('lambda_mean'):.6f} "
          f"hashprobe={summary['hash_offset_probe']}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["r0_performance", "deterministic"])
    ap.add_argument("--datasets", nargs="+", default=["MIMIC", "NHANES"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--repeat", type=int, default=0,
                    help="repeat index; >0 writes to a separate dir for within-condition spread")
    ap.add_argument("--outroot", default=str(CODE_ROOT / "results" / "T1.3_equivalence"))
    args = ap.parse_args()

    outroot = Path(args.outroot)
    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            rows.append(run_one(ds, s, args.condition, outroot, repeat=args.repeat))

    df = pd.DataFrame(rows)
    agg = outroot / f"summary_{args.condition}{'_r%d' % args.repeat if args.repeat else ''}.csv"
    df.to_csv(agg, index=False)
    print(f"\n[DONE] {len(rows)} runs -> {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
