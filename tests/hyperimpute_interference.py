"""T2d.2: does HyperImpute's CUDA context interfere with the GPU queue?

HyperImpute is on the CPU queue, but the T2c.1 sweep found it allocating roughly
440 MiB of GPU memory per process -- 1326 / 2649 / 5297 / 7060 MiB at 3 / 6 / 12 /
16 workers, against a flat 3 MiB for KNN in the same sweep. Nothing in the code
says it should: `registry.py:26-32` discusses fit/transform separation, not the
device.

Why it matters: the schedule assumes the two queues run concurrently at about
1 % cost, and the evidence for that is the T2.0(c) gate -- **one** GPU job beside
**one** CPU job, n=10. It does not cover twelve HyperImpute processes each
holding a CUDA context while the GPU queue works. B81 measured 172x for two
contending CUDA contexts, so if this interferes, the CPU queue drags the GPU
queue to a standstill for the whole grid.

Design: three timed conditions, each measuring the same TabCSDI cell.

    A  TabCSDI alone                                  -- the baseline
    B  TabCSDI + 12 HyperImpute (as scheduled today)  -- the question
    C  TabCSDI + 12 HyperImpute with CUDA hidden      -- the cleanest fix, if needed

C sets `CUDA_VISIBLE_DEVICES=""` for the HyperImpute workers only. If B is slow
and C is not, the remedy is one environment variable rather than a queue change,
and P2d ranks that highest of the three options for exactly that reason.

    env PYTHONHASHSEED=2025 python tests/hyperimpute_interference.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2d_device"

GPU_CELL = ("TabCSDI", "MIMIC")
CPU_CELL = ("HyperImpute", "MIMIC")
N_CPU = 12
ELAPSED = re.compile(r"^ELAPSED ([\d.]+)$", re.M)


def _src(method: str, dataset: str) -> str:
    return f"""
import os, sys, time
sys.path.insert(0, "{CODE_ROOT}")
import numpy as np, pandas as pd, yaml
import torch
from baselines.schema import DataSchema
from baselines.registry import build_baseline_imputer
from common import determinism
from experiments.run_grid import _apply_training_protocol
R = "{CODE_ROOT}"
complete = pd.read_csv(f"{{R}}/data/derived_shuffled/{dataset}_complete.csv")
mask = np.load(f"{{R}}/data/masks/clinical_v1_shuffled/{dataset}/"
               f"{dataset}_MAR_30per_mask.npy").astype(bool)
schema = DataSchema.from_yaml(f"{{R}}/configs/datasets.yaml", "{dataset}")
feats = list(schema.categorical_vars) + list(schema.continuous_vars)
mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
missing = complete[feats].mask(mask_df)
determinism.apply("deterministic", seed=1)
imp = build_baseline_imputer("{method}",
        categorical_vars=list(schema.categorical_vars),
        continuous_vars=list(schema.continuous_vars),
        seed=1, use_gpu={method == "TabCSDI"})
_apply_training_protocol("{method}", imp, False)
t0 = time.time()
imp.impute(missing, schema)
print(f"ELAPSED {{time.time()-t0:.2f}}", flush=True)
"""


def _gpu_mem() -> int:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int(q.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def condition(label: str, n_cpu: int, hide_cuda: bool) -> dict:
    """Time one TabCSDI cell with n_cpu HyperImpute jobs beside it."""
    env = dict(os.environ)
    cpu_env = dict(env)
    if hide_cuda:
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""

    cpu_procs = []
    for _ in range(n_cpu):
        cpu_procs.append(subprocess.Popen(
            [sys.executable, "-c", _src(*CPU_CELL)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=cpu_env))
    time.sleep(20)          # let the CPU side reach steady state first
    mem_before = _gpu_mem()

    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", _src(*GPU_CELL)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    out, err = p.communicate()
    wall = time.time() - t0
    mem_peak = _gpu_mem()

    for q in cpu_procs:
        if q.poll() is None:
            q.kill()
    for q in cpu_procs:
        q.wait()

    g = ELAPSED.search(out or "")
    rec = {"condition": label, "n_hyperimpute": n_cpu,
           "cuda_hidden_from_cpu_workers": hide_cuda,
           "tabcsdi_wall_sec": round(wall, 1),
           "tabcsdi_impute_sec": float(g.group(1)) if g else None,
           "gpu_mem_before_mib": mem_before, "gpu_mem_peak_mib": mem_peak,
           "gpu_job_ok": bool(g)}
    if not g:
        rec["error"] = f"rc={p.returncode} " + (err or "").strip()[-300:]
    print(f"[{label}] TabCSDI {rec['tabcsdi_impute_sec']}s "
          f"(wall {rec['tabcsdi_wall_sec']}s)  "
          f"gpu_mem {mem_before}->{mem_peak} MiB  ok={rec['gpu_job_ok']}",
          flush=True)
    if not g:
        print(f"    {rec['error']}", flush=True)
    return rec


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [condition("A_alone", 0, False),
            condition("B_with_12_hyperimpute", N_CPU, False),
            condition("C_with_12_hyperimpute_cuda_hidden", N_CPU, True)]

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hyperimpute_interference.csv", index=False)
    print("\n" + df.to_string(index=False))

    base = df.loc[df.condition == "A_alone", "tabcsdi_impute_sec"].iloc[0]
    if base is None:
        print("\nFAILED: the baseline run did not complete.", file=sys.stderr)
        return 1
    print(f"\nbaseline (TabCSDI alone): {base:.1f}s")
    verdict = 0
    for _, r in df[df.condition != "A_alone"].iterrows():
        if not r.gpu_job_ok:
            print(f"  {r.condition}: GPU job DID NOT COMPLETE -> severe interference")
            verdict = 1
            continue
        ratio = r.tabcsdi_impute_sec / base
        tag = "no meaningful interference" if ratio <= 1.25 else "INTERFERENCE"
        print(f"  {r.condition}: {r.tabcsdi_impute_sec:.1f}s = {ratio:.2f}x -> {tag}")
        if ratio > 1.25 and not r.cuda_hidden_from_cpu_workers:
            verdict = 1
    print(f"\nwrote {OUT}/hyperimpute_interference.csv")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
