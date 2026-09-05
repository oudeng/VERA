"""T2d.1: does SNI belong on the GPU queue at all?

The question is forced by two measurements that do not fit together. SNI holds a
CUDA context but samples 0 % GPU utilisation on 60 of 60 samples at 584 MiB while
pinning a single core (B81); and `SNI x CDC2022` costs >= 34 000 s per cell, which
on 111 per-feature fits of 200 epochs is about **1.5 s per epoch** for a
`[112->256->128->64]` MLP over 3000 rows -- three orders of magnitude off what
that should cost. If the device is idle, putting the method on the serial GPU
queue buys nothing and costs everything: the CPU queue runs 12 wide.

The rule that decides this was written and committed before the first run, in
`docs/T2d1_decision_rule.md`. Nothing here may restate it loosely -- the verdict
is computed by `verdict()` from that file's three conditions and no others.

Modes:

    --mode single       one job at a time; gives the solo per-cell cost (C1)
    --mode concurrency  N jobs at once; gives the per-cell cost under load (C2)
    --mode protocol     early stopping vs forced 200 epochs, paired (P2d section 4)

Every mode records wall-clock, metrics, epochs actually run, and peak RSS, so a
run never has to be repeated to answer a question the harness could have logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2d_device"

#: The CUDA side of C1 is not re-run. The P2c smoke arm measured exactly this
#: configuration -- SNI, MAR@30 %, seed 1, 200 epochs, early stopping disabled --
#: on an otherwise idle card. Re-running it would cost 1.1 h and change nothing.
#: `SNI/AutoMPG` is deliberately absent: its smoke timing is void (B81).
CUDA_REFERENCE = {"MIMIC": 1952.0, "NHANES": 1895.7, "Concrete": 123.5,
                  "eICU": 980.4, "ComCri": 834.5}

RESULT = re.compile(r"^RESULT (\{.*\})$", re.M)


def _child_source(dataset: str, seed: int, use_gpu: bool, disable_es: bool,
                  n_rows: int | None) -> str:
    """Source for one SNI run. numpy before torch -- see B80."""
    return f"""
import os, sys, time, json, resource
sys.path.insert(0, "{CODE_ROOT}")
import numpy as np, pandas as pd, yaml
import torch
from baselines.schema import DataSchema
from common import determinism
from evaluation.metrics import evaluate_imputation
from sni.imputer import SNIConfig, SNIImputer

R = "{CODE_ROOT}"
complete = pd.read_csv(f"{{R}}/data/derived_shuffled/{dataset}_complete.csv")
mask = np.load(f"{{R}}/data/masks/clinical_v1_shuffled/{dataset}/"
               f"{dataset}_MAR_30per_mask.npy").astype(bool)
schema = DataSchema.from_yaml(f"{{R}}/configs/datasets.yaml", "{dataset}")
feats = list(schema.categorical_vars) + list(schema.continuous_vars)
mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]

n_rows = {n_rows if n_rows else 'None'}
if n_rows is not None:
    # Down-sample rows only; d is held fixed because R2-4 is about table width.
    # Taking the head rather than a random subset keeps it reproducible without
    # introducing a second seed into a timing measurement.
    complete = complete.iloc[:n_rows].reset_index(drop=True)
    mask_df = mask_df.iloc[:n_rows].reset_index(drop=True)

missing = complete[feats].mask(mask_df)
determinism.apply("deterministic", seed={seed})

imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                 continuous_vars=list(schema.continuous_vars),
                 config=SNIConfig(seed={seed}, use_gpu={use_gpu}))
proto = yaml.safe_load(open(f"{{R}}/configs/training_protocol.yaml"))["protocol"]
epochs = int(proto["epochs"]["SNI"])
imp.cfg.epochs = epochs
if {disable_es}:
    imp.cfg.early_stopping_patience = epochs + 1     # same trick as run_grid
t0 = time.time()
X = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
elapsed = time.time() - t0

res = evaluate_imputation(X_imputed=X, X_complete=complete[feats],
                          X_missing=missing,
                          categorical_vars=list(schema.categorical_vars),
                          continuous_vars=list(schema.continuous_vars),
                          mask_df=mask_df)
s = dict(res.summary) if hasattr(res, "summary") else dict(res)
out = {{"dataset": "{dataset}", "seed": {seed},
       "device": "cuda" if {use_gpu} else "cpu",
       "early_stopping_disabled": bool({disable_es}),
       "n_rows": int(len(complete)), "n_features": len(feats),
       "wall_sec": round(elapsed, 2),
       "peak_rss_mib": round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
       "cont_R2": s.get("cont_R2"), "cont_NRMSE": s.get("cont_NRMSE"),
       "epochs_run": int(getattr(imp, "_total_epochs_run", -1))}}
print("RESULT " + json.dumps(out), flush=True)
"""


def run_batch(specs, label: str) -> tuple:
    """Launch every spec at once; return (records, batch wall, n_failed)."""
    procs = []
    t0 = time.time()
    for kw in specs:
        procs.append(subprocess.Popen(
            [sys.executable, "-c", _child_source(**kw)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    recs, failed, notes = [], [], []
    for p in procs:
        out, err = p.communicate()
        g = RESULT.search(out or "")
        if g:
            recs.append(json.loads(g.group(1)))
            # Warnings from a *successful* run were previously dropped on the
            # floor. That cost real time in T2d.1: the question "did CUDA use the
            # non-deterministic memory-efficient attention kernel?" was answerable
            # from these lines, but the logs were empty, so it had to be settled
            # by a separate experiment. Same class as B80 -- discarding
            # diagnostics because the exit code looked fine.
            for line in (err or "").splitlines():
                if "Warning" in line or "warn" in line.lower():
                    notes.append(line.strip()[:200])
        else:
            failed.append(f"rc={p.returncode} {(err or '').strip()[-400:]}")
    wall = time.time() - t0
    for f in failed:
        print(f"  CHILD FAILED: {f}", flush=True)
    for n in sorted(set(notes)):
        print(f"  child warning: {n}", flush=True)
    return recs, wall, len(failed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "concurrency", "protocol"],
                    required=True)
    ap.add_argument("--datasets", nargs="*", default=["Concrete", "NHANES", "MIMIC"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--rows", type=int, default=None,
                    help="down-sample to this many rows (d unchanged)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    use_gpu = a.device == "cuda"
    rows = []

    if a.mode == "single":
        for ds in a.datasets:
            for seed in a.seeds:
                recs, wall, nf = run_batch([dict(dataset=ds, seed=seed,
                                                 use_gpu=use_gpu, disable_es=True,
                                                 n_rows=a.rows)], f"{ds}")
                for r in recs:
                    r.update(concurrency=1, batch_wall_sec=round(wall, 1),
                             sec_per_cell=round(wall, 1))
                    ref = CUDA_REFERENCE.get(ds)
                    r["cuda_reference_sec"] = ref
                    r["cpu_over_cuda"] = (round(r["wall_sec"] / ref, 2)
                                          if ref and not use_gpu else None)
                    print(f"[{ds}/{a.device}/s{seed}] {r['wall_sec']:8.1f}s  "
                          f"R2={r['cont_R2']:8.4f}  "
                          f"rss={r['peak_rss_mib']:.0f}MiB"
                          + (f"  vs cuda {ref:.1f}s = {r['cpu_over_cuda']}x"
                             if r["cpu_over_cuda"] else ""), flush=True)
                rows.extend(recs)
                if nf:
                    print(f"  {nf} failure(s) on {ds}", flush=True)

    elif a.mode == "concurrency":
        ds = a.datasets[0]
        for n in [1, a.concurrency]:
            specs = [dict(dataset=ds, seed=1 + i, use_gpu=use_gpu,
                          disable_es=True, n_rows=a.rows) for i in range(n)]
            recs, wall, nf = run_batch(specs, f"{ds}x{n}")
            per_cell = wall / n
            ok = len(recs) == n
            print(f"[{ds}/{a.device}] conc={n:<3d} wall={wall:8.1f}s  "
                  f"per-cell={per_cell:7.1f}s  completed={len(recs)}/{n}  "
                  f"peak_rss_max={max((r['peak_rss_mib'] for r in recs), default=0):.0f}MiB",
                  flush=True)
            for r in recs:
                r.update(concurrency=n, batch_wall_sec=round(wall, 1),
                         sec_per_cell=round(per_cell, 1),
                         all_completed=ok)
            rows.extend(recs)

    else:   # protocol
        for ds in a.datasets:
            for seed in a.seeds:
                # Paired by data, mask and seed -- NOT by running at the same
                # time. An earlier version launched the two arms as one batch of
                # two "so they share machine state"; on CUDA that is two
                # concurrent GPU jobs, i.e. the 172x penalty of B81 applied to
                # the very measurement meant to quantify a protocol change.
                # Sequential is both correct and, on this hardware, faster.
                recs = []
                for disable in (False, True):
                    got, wall, nf = run_batch([dict(dataset=ds, seed=seed,
                                                    use_gpu=use_gpu,
                                                    disable_es=disable,
                                                    n_rows=a.rows)], f"{ds}s{seed}")
                    for r in got:
                        r.update(concurrency=1, batch_wall_sec=round(wall, 1))
                    recs.extend(got)
                rows.extend(recs)
                by = {r["early_stopping_disabled"]: r for r in recs}
                if False in by and True in by:
                    d = by[True]["cont_R2"] - by[False]["cont_R2"]
                    print(f"[{ds} s{seed}] early-stop R2={by[False]['cont_R2']:8.4f}"
                          f"  forced-200 R2={by[True]['cont_R2']:8.4f}"
                          f"  delta={d:+.4f}", flush=True)

    if not rows:
        print("\nFAILED: no run completed. Nothing was measured.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    tag = a.tag or f"{a.mode}_{a.device}"
    path = OUT / f"{tag}.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
