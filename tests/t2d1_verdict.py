"""Compute the T2d.1 verdict mechanically from the rule committed beforehand.

The rule is `docs/T2d1_decision_rule.md`, committed at 686702f before the first
run. This file implements those three conditions and nothing else. It exists so
the verdict is not reached by eye: "should SNI stay on the GPU" has been answered
wrongly once already, and the failure mode there was a plausible argument, not a
bad measurement.

Each condition prints its own inputs, so a reader can check the arithmetic
against the rule file without rerunning anything.

    env PYTHONHASHSEED=2025 python tests/t2d1_verdict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2d_device"

C1_DATASETS = ["MIMIC", "NHANES", "Concrete"]
C1_MAX_RATIO = 3.0
C2_MAX_RATIO = 1.5
C3_ABS_FALLBACK = 1e-3          # used only when cross-seed spread is degenerate


def _load(name: str) -> pd.DataFrame | None:
    p = OUT / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def c1() -> tuple[bool | None, list]:
    """CPU per-cell cost <= 3.0x CUDA, on every one of the three datasets."""
    df = _load("C1_single_cpu")
    if df is None:
        return None, [("C1", "no data", "results/T2d_device/C1_single_cpu.csv missing")]
    rows, ok = [], True
    for ds in C1_DATASETS:
        sub = df[df.dataset == ds]
        if sub.empty:
            rows.append((ds, None, None, "MISSING"))
            ok = False
            continue
        r = sub.iloc[0]
        ratio = r.wall_sec / r.cuda_reference_sec
        passed = ratio <= C1_MAX_RATIO
        ok &= bool(passed)
        rows.append((ds, round(r.wall_sec, 1), round(r.cuda_reference_sec, 1),
                     f"{ratio:.2f}x {'PASS' if passed else 'FAIL'}"))
    return ok, rows


def c2() -> tuple[bool | None, list]:
    """Per-cell cost at 12-way <= 1.5x solo, counting only complete batches."""
    rows, ok, seen = [], True, False
    for tag in ("C2_conc_concrete", "C2_conc_mimic"):
        df = _load(tag)
        if df is None:
            continue
        seen = True
        ds = df.dataset.iloc[0]
        by = df.groupby("concurrency").agg(
            per_cell=("sec_per_cell", "first"),
            n=("dataset", "size"),
            complete=("all_completed", "first")).reset_index()
        solo = by[by.concurrency == 1]
        wide = by[by.concurrency > 1]
        if solo.empty or wide.empty:
            rows.append((ds, "incomplete sweep", "", "FAIL"))
            ok = False
            continue
        s = solo.per_cell.iloc[0]
        for _, w in wide.iterrows():
            # A batch where not every job finished fails outright: that is
            # exactly how the GPU queue failed, and averaging over the jobs that
            # did finish would have hidden it.
            if not bool(w.complete):
                rows.append((ds, f"conc {int(w.concurrency)}",
                             "not all jobs completed", "FAIL"))
                ok = False
                continue
            ratio = w.per_cell / s
            passed = ratio <= C2_MAX_RATIO
            ok &= bool(passed)
            rows.append((ds, f"solo {s:.1f}s",
                         f"conc {int(w.concurrency)} {w.per_cell:.1f}s",
                         f"{ratio:.2f}x {'PASS' if passed else 'FAIL'}"))
    return (ok if seen else None), rows


def c3() -> tuple[bool | None, list]:
    """|CPU-CUDA| at seed 1 <= the metric's own cross-seed spread on CUDA."""
    cpu = _load("C1_single_cpu")
    proto = _load("protocol_pairs")
    if cpu is None or proto is None:
        return None, [("C3", "no data",
                       "needs C1_single_cpu.csv and protocol_pairs.csv")]
    rows, ok = [], True
    forced = proto[proto.early_stopping_disabled]
    for ds in C1_DATASETS:
        s = forced[forced.dataset == ds]
        c = cpu[cpu.dataset == ds]
        if s.empty or c.empty:
            rows.append((ds, "MISSING", "", "FAIL"))
            ok = False
            continue
        for metric in ("cont_R2", "cont_NRMSE"):
            spread = float(s[metric].max() - s[metric].min())
            # Seed 1 on CUDA comes from the protocol pairs (same configuration).
            ref = s[s.seed == 1]
            if ref.empty:
                rows.append((ds, metric, "no seed-1 CUDA row", "FAIL"))
                ok = False
                continue
            diff = abs(float(c.iloc[0][metric]) - float(ref.iloc[0][metric]))
            tol = spread if spread > C3_ABS_FALLBACK else C3_ABS_FALLBACK
            passed = diff <= tol
            ok &= bool(passed)
            rows.append((ds, metric, f"|d|={diff:.5f} vs tol {tol:.5f}"
                         + (" (abs fallback)" if spread <= C3_ABS_FALLBACK else
                            f" (seed spread over {len(s)} seeds)"),
                         "PASS" if passed else "FAIL"))
    return ok, rows


def main() -> int:
    print("T2d.1 verdict — computed from docs/T2d1_decision_rule.md (committed "
          "686702f, before any run)\n")
    results = {}
    for name, fn, desc in (("C1", c1, f"CPU per-cell cost <= {C1_MAX_RATIO}x CUDA, every dataset"),
                           ("C2", c2, f"per-cell at 12-way <= {C2_MAX_RATIO}x solo, all jobs complete"),
                           ("C3", c3, "CPU/CUDA metric gap <= cross-seed spread")):
        verdict, rows = fn()
        results[name] = verdict
        state = {True: "PASS", False: "FAIL", None: "NOT YET MEASURED"}[verdict]
        print(f"{name}  {desc}\n     -> {state}")
        for r in rows:
            print("       " + "  ".join(str(x) for x in r))
        print()

    if any(v is None for v in results.values()):
        print("VERDICT: incomplete — one or more conditions have no data yet.")
        return 2
    if all(results.values()):
        print("VERDICT: PASS -> SNI moves to the CPU queue; grid stays at 2565 cells.")
        return 0
    failed = [k for k, v in results.items() if not v]
    print(f"VERDICT: FAIL on {', '.join(failed)} -> fall back in the order fixed "
          f"by the rule: (1) down-sample CDC2022 keeping d=41, (2) MAR@30% only "
          f"on CDC2022, (3) seeds last.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
