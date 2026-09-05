"""Recompute the fair same-host recovery result from the per-cell values.

Self-contained on purpose: it reads one CSV and re-derives every number the
manuscript quotes for that comparison, so a reader holding only the package
can check +0.159 without trusting the summary JSON that sits beside it.

    PYTHONHASHSEED=2025 python code/experiments/recompute_fair_pair.py

It writes nothing. Run it from an unpacked package directory (it finds the
CSV under evidence/), or point it at a file with --cells.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: The comparison the manuscript reports, and its oracle-caliber control.
PAIRS = [("Permutation-on-SNI-fair-noOracle", "SNI-D-fairhost",
          "symmetric error signal (the manuscript's reading)"),
         ("Permutation-on-SNI-fair-oracle", "SNI-D-fairhost",
          "withheld-truth error signal (control)")]
#: The archived procedure, mirrored exactly so this script and the shipped
#: JSON agree to the printed digit: resample the SEED-level medians (never
#: cells within a seed), 10,000 draws, RNG seeded 20260831.
BOOT_SEED = 20260831
N_BOOT = 10000


def blocks(cells: pd.DataFrame, a: str, b: str) -> dict:
    """Per-seed arrays of the within-host paired difference a - b.

    The seed is the inference unit; the three regimes sit inside it. Treating
    the fifteen regime x seed cells as fifteen independent pairs is the
    pseudo-replication this revision corrects, so it is not done here.
    """
    piv = cells.pivot_table(index=["regime", "seed"], columns="method",
                            values="auroc")
    for col in (a, b):
        if col not in piv.columns:
            raise SystemExit(f"the cell file has no column {col!r}; it holds "
                             f"{sorted(piv.columns)}")
    d = (piv[a] - piv[b]).dropna()
    return {int(s): d.xs(s, level="seed").to_numpy(float)
            for s in sorted({s for _r, s in d.index})}


def seed_block_T(bl: dict) -> float:
    """The estimand used throughout: the mean of the seed-level medians."""
    return float(np.mean([float(np.median(v)) for v in bl.values()]))


def boot_ci(bl: dict) -> list:
    """Seed-only bootstrap: resample seeds, never cells within a seed."""
    rng = np.random.default_rng(BOOT_SEED)
    meds = np.array([float(np.median(np.asarray(v, float)))
                     for v in bl.values()])
    B = len(meds)
    stats = np.mean(meds[rng.integers(0, B, (N_BOOT, B))], axis=1)
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def sign_exact(bl: dict) -> dict:
    """Exact two-sided sign enumeration over seed blocks.

    With five seeds the smallest attainable two-sided p is 2/2^5 = 0.0625;
    that floor is why the comparison is reported as inconclusive rather than
    as a win.
    """
    obs = seed_block_T(bl)
    keys = list(bl)
    stats = []
    for flips in itertools.product([1, -1], repeat=len(keys)):
        stats.append(seed_block_T({k: f * bl[k] for k, f in zip(keys, flips)}))
    stats = np.asarray(stats)
    p = float((np.abs(stats) >= abs(obs) - 1e-12).mean())
    return {"p_two_sided": p, "floor_two_sided": 2.0 / 2 ** len(keys),
            "n_blocks": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    cand = [Path(a.cells)] if a.cells else [
        here.parent / "evidence" / "fair_same_host_recovery_cells.csv",
        here / "fair_same_host_recovery_cells.csv",
        Path("evidence/fair_same_host_recovery_cells.csv"),
        # ... and where the live repository keeps it, so the script a reviewer
        # runs from the package is the same one that runs here.
        here.parent / "results" / "T6_symmetry"
        / "fair_same_host_recovery_cells.csv",
    ]
    f = next((c for c in cand if c.exists()), None)
    if f is None:
        raise SystemExit(f"cell file not found; looked in {[str(c) for c in cand]}")
    cells = pd.read_csv(f)
    print(f"cells: {f}  ({len(cells)} rows, "
          f"{cells.method.nunique()} objects, {cells.seed.nunique()} seeds, "
          f"{cells.regime.nunique()} regimes)")
    out = {}
    for a_, b_, what in PAIRS:
        bl = blocks(cells, a_, b_)
        T = seed_block_T(bl)
        ci = boot_ci(bl)
        sg = sign_exact(bl)
        pos = sum(int((v > 0).sum()) for v in bl.values())
        n = sum(len(v) for v in bl.values())
        out[a_] = {"T": T, "ci95": ci, **sg, "cells_favouring_first": pos,
                   "cells_total": n}
        print(f"\n{what}\n  {a_} - {b_}")
        print(f"  T (mean of seed medians) = {T:+.6f}   -> {T:+.3f}")
        print(f"  seed-only bootstrap 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"  exact sign enumeration p = {sg['p_two_sided']:.4f} "
              f"(attainable floor {sg['floor_two_sided']:.4f}, "
              f"{sg['n_blocks']} seed blocks)")
        print(f"  cells favoring the first object: {pos}/{n}")
    d = out["Permutation-on-SNI-fair-noOracle"]["T"]
    o = out["Permutation-on-SNI-fair-oracle"]["T"]
    print(f"\nallowing the withheld true values changed T by {o - d:+.6f} "
          f"({100 * (o - d) / o:.1f}% of the withheld-truth effect), "
          f"from {d:+.3f} to {o:+.3f}")
    print("\n" + json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
