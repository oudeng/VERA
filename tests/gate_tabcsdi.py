"""T2.0(b) TabCSDI reproduction gate.

P1 section 4.3 ran TabCSDI once, on CPU, after misdiagnosing a GPU fault, and the
legacy arm did not reproduce R0's published numbers (MIMIC NRMSE 0.0859 against
0.0640; NHANES 0.522 against 0.215). A broken baseline contaminates every row of
Table 1, so this must be settled before the full grid.

This gate re-runs TabCSDI **on the GPU**, five seeds, through the legacy
(``X_complete``-consuming) interface, so it is a like-for-like comparison with
what R0 published.

Reading the result requires care: R0's own TabCSDI is wildly unstable on NHANES.
Its five published seeds give R-squared of +0.16, +0.01, -8.7, -19.3 and -224.0,
NRMSE 0.109 to 0.347 and MAE 11.4 to 122.4. The gate therefore asks whether our
re-run lands inside R0's own cross-seed spread, not whether it matches a point
estimate -- there is no stable point estimate to match.
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_EQ_THREADS", "4")
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

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
R0_TREE = CODE_ROOT.parent / "project_sni_R0" / "sni"

from baselines.registry import build_baseline_imputer      # noqa: E402
from baselines.schema import DataSchema                    # noqa: E402
from sni.dataio import load_complete_and_missing, cast_dataframe_to_schema  # noqa: E402
from sni.metrics import evaluate_imputation                # noqa: E402
from common import runconfig                               # noqa: E402

# Dataclass defaults from R0 baselines/registry.py TabCSDIBaseline; R0's
# budget_params was empty for this method, so the defaults are what ran.
TABCSDI_HP = dict(diffusion_steps=50, n_samples=10, d_model=128, n_heads=4,
                  n_layers=3, epochs=200, batch_size=64, lr=1e-3)

DATASETS = {
    "MIMIC": dict(complete="data/MIMIC_complete.csv",
                  missing="data/MIMIC/MIMIC_MAR_30per.csv",
                  categorical=["SpO2", "ALARM"],
                  continuous=["RESP", "ABP", "SBP", "DBP", "HR", "PULSE"]),
    "NHANES": dict(complete="data/NHANES_complete.csv",
                   missing="data/NHANES/NHANES_MAR_30per.csv",
                   categorical=["gender_std", "age_band"],
                   continuous=["waist_circumference", "systolic_bp", "diastolic_bp",
                               "triglycerides", "hdl_cholesterol", "fasting_glucose",
                               "age", "bmi", "hba1c", "metabolic_score"]),
}


def run_one(dataset: str, seed: int, use_gpu: bool, mode: str, outroot: Path) -> dict:
    spec = DATASETS[dataset]
    cat, cont = spec["categorical"], spec["continuous"]

    X_complete, X_missing, schema_sni = load_complete_and_missing(
        input_complete=str(R0_TREE / spec["complete"]),
        input_missing=str(R0_TREE / spec["missing"]),
        categorical_vars=cat, continuous_vars=cont)
    mask_df = X_missing.isna()

    legacy = (mode == "legacy")
    imp = build_baseline_imputer("TabCSDI", categorical_vars=cat,
                                 continuous_vars=cont, seed=seed,
                                 use_gpu=use_gpu, legacy_oracle=legacy,
                                 **TABCSDI_HP)
    schema = DataSchema.from_var_lists(categorical_vars=cat, continuous_vars=cont,
                                       dataset=dataset)

    t0 = time.time()
    X_imp = imp.run(X_missing, schema=schema,
                    X_complete=X_complete if legacy else None)
    runtime = time.time() - t0

    X_imp = cast_dataframe_to_schema(X_imp, schema_sni)
    res = evaluate_imputation(X_imputed=X_imp, X_complete=X_complete,
                             X_missing=X_missing, categorical_vars=cat,
                             continuous_vars=cont, mask_df=mask_df)
    s = dict(res.summary)
    s.update(dataset=dataset, seed=seed, mode=mode,
             device="gpu" if use_gpu else "cpu", runtime_sec=runtime)

    outdir = outroot / f"{'gpu' if use_gpu else 'cpu'}_{mode}" / f"{dataset}_s{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics_summary.json").write_text(json.dumps(s, indent=2, default=str))
    runconfig.write(outdir, runconfig.build(
        exp_id=f"TabCSDI_{dataset}_s{seed}_{mode}", method="TabCSDI",
        params={**TABCSDI_HP, "use_gpu": use_gpu, "legacy_oracle": legacy},
        inputs={"complete": str(R0_TREE / spec["complete"]),
                "missing": str(R0_TREE / spec["missing"])},
        seeds={"model": seed}, determinism={"mode": "as-is (baseline)"},
        extra={"purpose": "P2/T2.0(b) TabCSDI reproduction gate"}))

    print(f"[OK] TabCSDI {dataset} s{seed} {mode} "
          f"{'gpu' if use_gpu else 'cpu'}  {runtime:6.1f}s  "
          f"NRMSE={s['cont_NRMSE']:.6f} MAE={s['cont_MAE']:.4f} "
          f"R2={s['cont_R2']:.4f} Acc={s['cat_Accuracy']:.4f}", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MIMIC", "NHANES"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--mode", default="legacy", choices=["legacy", "deleaked"])
    ap.add_argument("--outroot", default=str(CODE_ROOT / "results" / "T2.0_gate" / "tabcsdi"))
    a = ap.parse_args()

    rows = [run_one(ds, s, a.device == "gpu", a.mode, Path(a.outroot))
            for ds in a.datasets for s in a.seeds]
    out = Path(a.outroot) / f"summary_{a.device}_{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[DONE] {len(rows)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
