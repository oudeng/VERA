"""Fit SNI's cost on CDC2022 against row count, and say how much to trust it.

`SNI x CDC2022` is the one cell in the grid with no measurement -- it was stopped
after 9 h 26 m without finishing. Three cost models have already been applied to
it and all three failed badly (by >=18x, >=6.8x and >=3.4x), which is why the
committed rule (`docs/T2d1_decision_rule.md`) requires any estimate here to carry
its uncertainty rather than arrive as a bare number.

So this fits a power law on measured points at n = 500, 1000, 1500 with d held at
41, reports the exponent with a confidence interval, and -- if the full n = 3000
run completed -- checks the prediction against it. That last step is the only
thing that turns the curve from a guess into a validated instrument, and it is
reported either way.

    env PYTHONHASHSEED=2025 python stats/cdc_scaling.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2d_device"
FULL_N = 3000
GRID_CELLS = 15          # SNI x CDC2022 under the slimmed grid


def load() -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(OUT / "cdc_*.csv")):
        d = pd.read_csv(f)
        d["src"] = Path(f).stem
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d[d.dataset == "CDC2022"].sort_values("n_rows")


def fit(sub: pd.DataFrame):
    """Power law t = a * n^b, fitted in log space. Returns (a, b, se_b, r2)."""
    x = np.log(sub.n_rows.to_numpy(float))
    y = np.log(sub.wall_sec.to_numpy(float))
    n = len(x)
    b, loga = np.polyfit(x, y, 1)
    pred = loga + b * x
    resid = y - pred
    if n > 2:
        s2 = float(resid @ resid) / (n - 2)
        se_b = float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))
    else:
        se_b = float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(loga)), float(b), se_b, r2


def main() -> int:
    d = load()
    if d.empty:
        print("no CDC2022 runs yet in results/T2d_device/", file=sys.stderr)
        return 2

    print(d[["n_rows", "wall_sec", "cont_R2", "peak_rss_mib", "src"]]
          .to_string(index=False))

    curve = d[d.n_rows < FULL_N]
    full = d[d.n_rows == FULL_N]

    if len(curve) < 2:
        print("\nneed at least two sub-sampled points to fit", file=sys.stderr)
        return 2

    a, b, se_b, r2 = fit(curve)
    print(f"\npower-law fit on {len(curve)} sub-sampled points "
          f"(d fixed at 41):")
    print(f"  t = {a:.4g} * n^{b:.3f}"
          + (f"   (exponent SE {se_b:.3f}, 95% CI "
             f"[{b - 1.96 * se_b:.3f}, {b + 1.96 * se_b:.3f}])"
             if np.isfinite(se_b) else "   (2 points: no error bar available)"))
    print(f"  log-log R^2 = {r2:.5f}")

    pred = a * FULL_N ** b
    if np.isfinite(se_b):
        lo = a * FULL_N ** (b - 1.96 * se_b)
        hi = a * FULL_N ** (b + 1.96 * se_b)
        print(f"\n  extrapolated to n={FULL_N}: {pred:.0f} s "
              f"(95% CI {lo:.0f}-{hi:.0f} s, i.e. "
              f"{lo / 3600 * GRID_CELLS:.1f}-{hi / 3600 * GRID_CELLS:.1f} h "
              f"over {GRID_CELLS} cells)")
    else:
        print(f"\n  extrapolated to n={FULL_N}: {pred:.0f} s "
              f"= {pred / 3600 * GRID_CELLS:.1f} h over {GRID_CELLS} cells "
              f"(NO error bar -- two points cannot give one)")

    if len(full):
        actual = float(full.wall_sec.iloc[0])
        err = (actual - pred) / pred
        print(f"\n  MEASURED at n={FULL_N}: {actual:.0f} s "
              f"= {actual / 3600 * GRID_CELLS:.1f} h over {GRID_CELLS} cells")
        print(f"  the curve was off by {err:+.1%} -- "
              + ("the curve is validated and may be reused"
                 if abs(err) <= 0.15 else
                 "**the curve is NOT validated**; use the measurement, and do "
                 "not reuse this fit elsewhere"))
        print(f"\n  USE: {actual / 3600 * GRID_CELLS:.1f} h (measured)")
    else:
        print(f"\n  n={FULL_N} did not complete; the extrapolation above is the "
              f"best available and its interval must be quoted with it.")
        print("  Note for the record: the previous three estimates of this cell "
              "were 1914.9 s, 5010 s and 9945 s per cell, against a measured "
              "lower bound of 34000 s. A point estimate here would not be "
              "credible without the interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
