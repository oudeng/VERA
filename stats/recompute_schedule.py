"""P2c section 6: recompute the grid schedule from measured runtimes.

The estimates in `configs/scheduling.yaml` were extrapolated from R0's runs on
R0's tables, by n and d. Four of the seven tables changed in R1, and the smoke
test has now measured every (method, dataset) on the tables we actually ship.
Two of those measurements move the answer:

  * KNN x CDC2022 -- 7657.9 s measured against 2200 s extrapolated (3.5x). The
    cause is implementation, not size: KNN loops over donor rows in Python
    (`baselines/KNN_v1.py:255-262`), about 9 million calls on a 3000x39 table.
  * KNN x MIMIC and HyperImpute x MIMIC -- roughly 2x, because the replacement
    MIMIC table is 2849x16 where R0's was 2052x8.

Elsewhere the extrapolations were pessimistic (median ratio 0.57).

Two adjustments are applied on top of the measurements:

  * the P2c protocol multiplier, since SNI and TabCSDI now train the configured
    200 epochs instead of stopping at 52-136;
  * the grid slimming, which cuts CDC2022 from 405 cells to 90.

Wall-clock assumes the queue policy in `configs/scheduling.yaml`: the GPU queue
serial, the CPU queue over N workers, the two running concurrently at a measured
cost of about 1 %.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"\[(\w+)\s*\]\s+(\S+)/(\S+)\s+R2=\s*\S+\s+NRMSE=\S+\s+([\d.]+)s")

#: How much more training the P2c protocol buys, per method. SNI stopped at
#: ~94 epochs per per-feature model and TabCSDI at 52-136; both now run 200.
#: Ratios are from `results/T2b_tabcsdi_budget/epoch_probe.csv`.
#:
#: Applied ONLY to extrapolated figures. The smoke test now runs under the P2c
#: protocol itself (`tests/smoke_nine_methods.py`), so a measured timing already
#: includes the extra training and multiplying it again would double-count. The
#: CPU-side measurements are unaffected either way: no CPU method has a stopping
#: rule.
PROTOCOL_MULTIPLIER = {"SNI": 200 / 94, "TabCSDI": 200 / 136}


#: Measurements that exist in the logs but must never be used as timings.
#: A wall-clock taken while something else held the GPU is not a measurement of
#: this cell; it is a measurement of contention. Left in the log (the metrics
#: from those runs are still valid -- contention changes timing, not numerics)
#: but excluded here, because silently ingesting one would put SNI x AutoMPG
#: into the schedule at 274 h instead of ~1.6 h.
VOID_TIMINGS = {
    ("SNI", "AutoMPG"): "ran 21969.2 s while an orphaned budget_sensitivity.py "
                        "shared the card (B81); 172x contention, not a timing. "
                        "Clean value comes from the T2c.1 conc-1 batch.",
}


#: Cells whose run was abandoned, leaving a lower bound rather than a timing.
#: Using the extrapolation for these would understate the grid by an order of
#: magnitude, so the bound is carried explicitly and labeled as a bound.
LOWER_BOUNDS = {
    ("SNI", "CDC2022"): (
        34000.0,
        "stopped after 9 h 26 m without finishing (first-author decision). "
        "37 imputable columns, 30 categorical -> 111 per-feature fits at ~112 "
        "one-hot input dims. The extrapolation of 1914.9 s/cell is low by at "
        "least 18x; every cost model disagreed with it and with each other."),
}


#: Clean timings sourced from somewhere other than the smoke log, used when the
#: smoke log's own value is void. Keyed the same way so they slot straight in.
OVERRIDES = {
    ("SNI", "AutoMPG"): (107.9,
                         "T2c.1 conc-1 batch, results/T2c_concurrency/"
                         "gpu_sweep.csv -- the uncontended value the voided "
                         "smoke run (21969.2 s) was supposed to provide"),
}


def measured_runtimes(verbose: bool = False) -> Dict[tuple, float]:
    out = {}
    for f in glob.glob(str(CODE_ROOT / "results" / "T2b_smoke" / "*.log")):
        for line in open(f):
            g = LINE.match(line)
            if g:
                key = (g.group(2), g.group(3))
                if key in VOID_TIMINGS:
                    if verbose:
                        print(f"  VOID  {key[0]}/{key[1]} "
                              f"({float(g.group(4)):.1f}s): {VOID_TIMINGS[key]}")
                    continue
                out[key] = float(g.group(4))
    for key, (sec, why) in OVERRIDES.items():
        out[key] = sec
        if verbose:
            print(f"  OVERRIDE {key[0]}/{key[1]} -> {sec:.1f}s: {why}")
    return out


def cell_counts() -> Dict[tuple, int]:
    """Cells per (method, dataset) under the slimmed grid."""
    import sys
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.run_grid import cells

    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    sched = yaml.safe_load((CODE_ROOT / "configs" / "scheduling.yaml").read_text())
    todo = cells(list(cfg["datasets"]), list(sched["method_placement"]), True)
    counts: Dict[tuple, int] = {}
    for m, d, _mech, _r, _s in todo:
        counts[(m, d)] = counts.get((m, d), 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-workers", type=int, default=3)
    ap.add_argument("--gpu-concurrency", type=int, default=1,
                    help="fallback for any GPU method without its own setting")
    # Per-method, because the two GPU methods are not alike. SNI sampled 0 % GPU
    # utilisation on 60 of 60 samples at 584 MiB while pinning one core -- it is
    # launch-latency bound and should widen freely. TabCSDI is one diffusion
    # model that does use the device. One number for both would be wrong for one
    # of them, and SNI is 82 % of the queue.
    ap.add_argument("--gpu-concurrency-sni", type=int, default=None)
    ap.add_argument("--gpu-concurrency-tabcsdi", type=int, default=None)
    a = ap.parse_args()

    sched = yaml.safe_load((CODE_ROOT / "configs" / "scheduling.yaml").read_text())
    est = sched["runtime_estimates_sec"]
    placement = sched["method_placement"]
    print("excluded measurements:")
    meas = measured_runtimes(verbose=True)
    for (m, d), (bound, why) in LOWER_BOUNDS.items():
        print(f"  BOUND {m}/{d} (>= {bound:.0f}s): {why}")
    counts = cell_counts()

    rows = []
    for (m, d), n in sorted(counts.items()):
        sec = meas.get((m, d))
        source = "measured" if sec is not None else "extrapolated"
        if sec is None:
            sec = float(est.get(m, {}).get(d, 0) or 0)
            sec *= PROTOCOL_MULTIPLIER.get(m, 1.0)   # extrapolation only
        if (m, d) in LOWER_BOUNDS:
            bound, _why = LOWER_BOUNDS[(m, d)]
            if bound > sec:
                sec, source = bound, "LOWER BOUND"
        rows.append({"method": m, "dataset": d, "queue": placement[m],
                     "cells": n, "sec_per_cell": round(sec, 1),
                     "hours": round(sec * n / 3600, 2), "source": source})
    df = pd.DataFrame(rows)

    out = CODE_ROOT / "results" / "T2c_concurrency"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "schedule_recomputed.csv", index=False)

    conc = {"SNI": a.gpu_concurrency_sni or a.gpu_concurrency,
            "TabCSDI": a.gpu_concurrency_tabcsdi or a.gpu_concurrency}

    gpu = df[df.queue == "gpu"]
    cpu_h = df[df.queue == "cpu"].hours.sum()
    # GPU methods modeled as consecutive phases, each at its own width. That is
    # the conservative reading: interleaving them could only help, and claiming
    # the help without having measured mixed-method contention would be exactly
    # the kind of unmeasured optimism this exercise exists to remove.
    gpu_h = gpu.hours.sum()
    gpu_wall = sum(g.hours.sum() / max(conc.get(m, a.gpu_concurrency), 1)
                   for m, g in gpu.groupby("method"))
    cpu_wall = cpu_h / max(a.cpu_workers, 1)
    wall = max(gpu_wall, cpu_wall) * 1.01      # measured co-residency cost ~1 %

    print(df.sort_values("hours", ascending=False).head(12).to_string(index=False))
    print(f"\n{int(df.cells.sum())} cells, "
          f"{int((df.source == 'measured').sum())}/{len(df)} (method, dataset) "
          f"pairs from measurement")
    print(f"  GPU queue : {gpu_h:6.1f} h total")
    for m, g in gpu.groupby("method"):
        c = max(conc.get(m, a.gpu_concurrency), 1)
        print(f"      {m:<10s} {g.hours.sum():6.1f} h / concurrency {c} "
              f"= {g.hours.sum()/c:6.1f} h")
    print(f"      {'-> GPU wall':<10s} {gpu_wall:6.1f} h")
    print(f"  CPU queue : {cpu_h:6.1f} h  / {a.cpu_workers} workers "
          f"= {cpu_wall:6.1f} h")
    print(f"  wall-clock lower bound: {wall:.1f} h = {wall/24:.2f} days "
          f"(critical path: {'GPU' if gpu_wall >= cpu_wall else 'CPU'})")
    if gpu_h > 100:
        print(f"\n  NOTE: GPU queue total {gpu_h:.1f} h exceeds the 100 h in P2c "
              f"section 7 condition 2. That condition is written on the queue "
              f"total, not on wall-clock; at concurrency it takes "
              f"{gpu_wall:.1f} h of wall-clock.")
    print(f"\nwrote {out/'schedule_recomputed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
