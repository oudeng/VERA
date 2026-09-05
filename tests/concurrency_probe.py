"""T2c.1: what concurrency do the two queues actually support?

The scheduling policy in `configs/scheduling.yaml` says one GPU job at a time,
and cites a P1 measurement: one process at 32 threads took 269 s per MIMIC run,
two processes at 8 threads each took over 6 hours. That is an **80x** penalty,
and for these model sizes -- per-feature MLPs and a small-table diffusion model
on a 48 GB card -- it is not credible as a GPU-contention number. It looks much
more like host-side CPU oversubscription, which would mean the real constraint
is thread count, not the GPU, and that 2-3 concurrent GPU jobs are fine.

The distinction is worth more than any other single scheduling decision: if the
GPU queue can run 2-3 wide, the 65 h critical path halves or thirds.

Since that paragraph was written the first half of the answer has been measured
directly, and it is not thread count either: during a live SNI run, GPU
utilisation was 0 % on 60 of 60 samples at a constant 584 MiB while the process
pinned a single core. SNI holds a CUDA context but its kernels are microseconds
long, so it is launch-latency bound -- host-side, as suspected, but for a
different reason than oversubscription. What remains to be measured is how far
that lets the queue widen, and whether TabCSDI behaves the same way (it should
not: one transformer-based diffusion model does use the device).

The CPU side needs measuring too, and for a different reason: KNN is a Python
double loop over donor rows (`baselines/KNN_v1.py:255-262`) and measured ~1.5 h
on CDC2022 against a 0.6 h extrapolation. The estimates in scheduling.yaml were
never measured on these tables.

Each batch reports wall-clock per cell *and* GPU utilisation, GPU memory and
host load (P2c section 3 asks for all four), because wall-clock alone cannot
tell a saturated device from an idle one. Nothing here writes into the results
tree beyond its own sweep CSV.

    env PYTHONHASHSEED=2025 python tests/concurrency_probe.py --queue gpu
    env PYTHONHASHSEED=2025 python tests/concurrency_probe.py --queue cpu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2c_concurrency"

#: GPU: **both** methods, swept separately rather than mixed.
#:
#: An earlier version of this file swept only TabCSDI, on the reasoning that one
#: cheap representative cell was enough. It is not. Under the recomputed schedule
#: SNI is 90.9 h of the GPU queue's 111.4 h, so it is SNI's concurrency behavior
#: that sets the critical path, and the two methods are not alike: sampled at
#: 1 Hz for 60 s during a live SNI run, GPU utilisation was 0 % on 60 of 60
#: samples at a constant 584 MiB, while the process held 99.6 % of a single core
#: across 44 threads. SNI is per-feature MLPs on a few thousand rows -- each
#: kernel is microseconds and the cost is Python and launch latency, not the
#: device. TabCSDI is one transformer-based diffusion model and genuinely uses
#: it. A concurrency limit measured on the second says nothing about the first.
#:
#: Swept separately, not mixed: a mixed pair measures cross-method interference,
#: which is not the quantity the queue policy needs.
#:
#: CPU: two groups, likewise swept separately, chosen for *contention
#: behavior* rather than for being slow.
#:
#: P2c section 3 names KNN x {CDC2022, NHANES, eICU} and a 3/4/6 sweep, under a
#: <=2 GPU-hour budget for T2c.1 as a whole. Those two constraints conflict:
#: KNN x CDC2022 measured **7657.9 s = 2.13 h** for a single cell, so one run of
#: it exhausts the budget before anything is swept. **KNN x eICU** (304.7 s) is
#: also on the instruction's list and is cheap enough to sweep five times, so it
#: stands in for the single-threaded majority; mixing the three named cells was
#: dropped because a mixed batch costs as much as its slowest member.
#:
#: The concurrency range is *extended* rather than substituted: 3/4/6 as
#: instructed, plus 12 and 16. The reason 3/4/6 alone is not enough is that this
#: host has **32 cores** and KNN is a pure-Python double loop over donor rows
#: (`baselines/KNN_v1.py:255-262`) with no internal threading, so the ceiling is
#: plausibly far above 6. `scheduling.yaml` currently assumes 3 workers; if 12 is
#: safe the CPU queue drops from ~29.7 h to ~7 h of wall-clock.
#:
#: HyperImpute x MIMIC (225.4 s) is added as a second group: it stands in for the
#: sklearn-threaded methods, which are where over-subscription would actually
#: bite -- that is the failure mode behind R0's 300 silent `baselines_deep`
#: deaths, so it gets measured rather than assumed.
GPU_CELLS = {"SNI": ("SNI", "AutoMPG"), "TabCSDI": ("TabCSDI", "AutoMPG")}
CPU_CELLS = {"KNN": [("KNN", "eICU")], "HyperImpute": [("HyperImpute", "MIMIC")]}
CPU_CONC = {"KNN": [3, 4, 6, 12, 16], "HyperImpute": [3, 6, 12, 16]}


def _one_run_cmd(method: str, dataset: str, threads: int) -> List[str]:
    return [sys.executable, "-c", f"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "{threads}"
os.environ["MKL_NUM_THREADS"] = "{threads}"
sys.path.insert(0, "{CODE_ROOT}")
# numpy BEFORE torch. Importing torch first loads libgomp, and mkl-service then
# refuses to initialize: "MKL_THREADING_LAYER=INTEL is incompatible with
# libgomp.so.1". Every child of the first sweep died on this in 0.3 s, and with
# stderr going to DEVNULL the sweep reported completed=0/N and exited 0.
import numpy as np, pandas as pd, yaml
import torch
torch.set_num_threads({threads})
from baselines.schema import DataSchema
from common import determinism
from experiments.run_grid import _build, _apply_training_protocol
R = "{CODE_ROOT}"
complete = pd.read_csv(f"{{R}}/data/derived_shuffled/{dataset}_complete.csv")
mask = np.load(f"{{R}}/data/masks/clinical_v1_shuffled/{dataset}/{dataset}_MAR_30per_mask.npy").astype(bool)
schema = DataSchema.from_yaml(f"{{R}}/configs/datasets.yaml", "{dataset}")
feats = list(schema.categorical_vars) + list(schema.continuous_vars)
mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
missing = complete[feats].mask(mask_df)
determinism.apply("deterministic", seed=1)
imp, is_sni = _build("{method}", schema, 1, {str(method in ("SNI", "TabCSDI"))})
_apply_training_protocol("{method}", imp, is_sni)
t0 = time.time()
X = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df) if is_sni else imp.impute(missing, schema)
print(f"ELAPSED {{time.time()-t0:.2f}}")
"""]


class _Sampler(threading.Thread):
    """Sample GPU utilisation, GPU memory and host load while a batch runs.

    P2c section 3 asks for "单格耗时、GPU 利用率与显存、CPU 负载" -- wall-clock
    alone cannot distinguish "the GPU is saturated" from "the GPU is idle and we
    are launch-bound", and that distinction is the whole point of the sweep.
    Sampled at 1 Hz; `nvidia-smi` failures are swallowed so a probe never dies of
    its own instrumentation.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.util: List[int] = []
        self.mem: List[int] = []
        self.load: List[float] = []
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                u, m = q.stdout.strip().splitlines()[0].split(", ")
                self.util.append(int(u))
                self.mem.append(int(m))
            except Exception:
                pass
            try:
                self.load.append(os.getloadavg()[0])
            except Exception:
                pass
            self._done.wait(1.0)

    def stop(self) -> dict:
        self._done.set()
        self.join(timeout=5)
        f = lambda xs, g: round(g(xs), 1) if xs else None
        return {"gpu_util_mean": f(self.util, lambda x: sum(x) / len(x)),
                "gpu_util_max": f(self.util, max),
                "gpu_util_nonzero_frac": (round(sum(1 for x in self.util if x > 0)
                                                / len(self.util), 3)
                                          if self.util else None),
                "gpu_mem_max_mib": f(self.mem, max),
                "load1_mean": f(self.load, lambda x: sum(x) / len(x)),
                "load1_max": f(self.load, max),
                "n_samples": len(self.util)}


#: A batch is abandoned once it exceeds this multiple of the same group's
#: concurrency-1 wall-clock. Without a cap the sweep is unaffordable: two
#: concurrent GPU jobs were measured at **172x** (B81), so a conc-2 batch of a
#: 128 s cell would run over six hours and a conc-3 batch longer still. A run
#: that hits the cap is not a failure to measure -- it establishes a lower bound
#: on the penalty, which is all the queue policy needs. Recorded as `capped`.
TIMEOUT_MULTIPLE = 10.0
MIN_TIMEOUT_SEC = 300.0


def sweep(cells, concurrencies, thread_counts, label: str) -> pd.DataFrame:
    rows = []
    baseline = None          # wall of this group's lowest-concurrency batch
    for threads in thread_counts:
        for n in concurrencies:
            picks = [cells[i % len(cells)] for i in range(n)]
            procs = []
            sampler = _Sampler()
            sampler.start()
            t0 = time.time()
            for method, dataset in picks:
                procs.append(subprocess.Popen(
                    _one_run_cmd(method, dataset, threads),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            deadline = (time.time() + max(MIN_TIMEOUT_SEC,
                                          TIMEOUT_MULTIPLE * baseline)
                        if baseline is not None else None)
            per, errs, capped = [], [], False
            for p in procs:
                try:
                    remaining = (deadline - time.time()) if deadline else None
                    if remaining is not None and remaining <= 0:
                        raise subprocess.TimeoutExpired(p.args, 0)
                    out, err = p.communicate(timeout=remaining)
                except subprocess.TimeoutExpired:
                    capped = True
                    for q in procs:
                        if q.poll() is None:
                            q.kill()
                    out, err = "", "abandoned at the contention cap"
                got = False
                for line in (out or "").splitlines():
                    if line.startswith("ELAPSED"):
                        per.append(float(line.split()[1]))
                        got = True
                if not got:
                    # stderr used to go to DEVNULL. A whole sweep then reported
                    # `completed=0/N` for every setting, exited 0, and left no
                    # trace of why -- twelve seconds of measuring nothing that
                    # looked like a successful run. Never discard the child's
                    # diagnostics.
                    errs.append(f"rc={p.returncode} " + (err or "").strip()[-500:])
            wall = time.time() - t0
            telemetry = sampler.stop()
            if baseline is None and len(per) == n:
                baseline = wall / n          # per-cell cost with no contention
            rec = {"queue": label, "threads": threads, "concurrency": n,
                   "capped": capped,
                   "slowdown_vs_conc1": (round(wall / n / baseline, 1)
                                         if baseline else None),
                   "wall_sec": round(wall, 1),
                   "per_run_mean": round(sum(per) / len(per), 1) if per else None,
                   "per_run_max": round(max(per), 1) if per else None,
                   "n_completed": len(per),
                   "sec_per_cell": round(wall / max(n, 1), 1),
                   **telemetry,
                   "cells": ",".join(f"{m}/{d}" for m, d in picks)}
            rows.append(rec)
            print(f"[{label}] threads={threads} conc={n}  wall={rec['wall_sec']}s  "
                  f"per-cell={rec['sec_per_cell']}s  completed={len(per)}/{n}  "
                  f"gpu_util={telemetry['gpu_util_mean']}% "
                  f"(nonzero {telemetry['gpu_util_nonzero_frac']}) "
                  f"gpu_mem={telemetry['gpu_mem_max_mib']}MiB "
                  f"load1={telemetry['load1_mean']}", flush=True)
            if capped:
                print(f"    CAPPED: abandoned at {wall:.0f}s "
                      f"(>= {rec['slowdown_vs_conc1']}x the conc-1 per-cell "
                      f"cost); the penalty is at least this large", flush=True)
            for e in errs[:2]:
                if "abandoned at the contention cap" not in e:
                    print(f"    CHILD FAILED: {e}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, nargs="*", default=None)
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before start (B48).",
              file=sys.stderr)
        return 2

    if a.queue == "gpu":
        conc = a.concurrency or [1, 2, 3]
        thr = a.threads or [1, 4, 8]
        groups = [(f"gpu:{m}", [cell]) for m, cell in GPU_CELLS.items()]
    else:
        # Two thread settings, not three: KNN has no internal threading, so a
        # third point buys nothing and costs a third of the sweep.
        thr = a.threads or [1, 8]
        groups = [(f"cpu:{m}", cells, a.concurrency or CPU_CONC[m])
                  for m, cells in CPU_CELLS.items()]

    if a.queue == "gpu":
        groups = [(lbl, cells, conc) for lbl, cells in groups]

    frames = []
    for label, cells, conc_for_group in groups:
        frames.append(sweep(cells, conc_for_group, thr, label))
    df = pd.concat(frames, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{a.queue}_sweep.csv", index=False)

    print("\n" + df.to_string(index=False))
    for label in df.queue.unique():
        sub = df[df.queue == label]
        ok = sub[sub.n_completed == sub.concurrency]
        if not len(ok):
            print(f"\n[{label}] no setting completed every run -- see n_completed")
            continue
        best = ok.loc[ok.sec_per_cell.idxmin()]
        base = ok[ok.concurrency == ok.concurrency.min()].sec_per_cell.min()
        print(f"\n[{label}] best throughput: concurrency={int(best.concurrency)} "
              f"threads={int(best.threads)} -> {best.sec_per_cell}s per cell "
              f"({base / best.sec_per_cell:.2f}x the lowest-concurrency setting)")
        # The number the schedule needs: does adding a second/third job cost more
        # than it buys? Reported per method, since they need not agree.
        for n in sorted(ok.concurrency.unique()):
            row = ok[ok.concurrency == n].sort_values("sec_per_cell").iloc[0]
            print(f"    concurrency {int(n)}: {row.sec_per_cell:8.1f} s/cell "
                  f"(per-run mean {row.per_run_mean:8.1f} s, threads {int(row.threads)})")
    print(f"\nwrote {OUT}/{a.queue}_sweep.csv")
    # A sweep in which nothing ran is not a result, and must not look like one.
    # The first run of this file reported `completed=0/N` on every setting and
    # still exited 0, so the chain moved on as if the queue had been measured.
    total_expected = int(df.concurrency.sum())
    total_done = int(df.n_completed.sum())
    if total_done == 0:
        print(f"\nFAILED: 0 of {total_expected} child runs completed. "
              f"Nothing was measured.", file=sys.stderr)
        return 1
    if total_done < total_expected:
        print(f"\nWARNING: only {total_done} of {total_expected} child runs "
              f"completed; settings with n_completed < concurrency are not "
              f"usable.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
