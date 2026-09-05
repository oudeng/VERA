"""T2c.3: budget sensitivity, for the ESM.

**This is no longer used to choose a budget.** P2c fixes the budget at the 200
epochs the code already configured, precisely so that no number gets selected
after the fact. What the scan is for now is disclosure: how far each method sits
from its own convergence point at 200 epochs, shown as a curve rather than
asserted.

That also answers "were the baselines treated fairly?" in the only way that
really settles it -- by showing the reader the whole budget-response curve for
every affected method, ours included.

Design (P2c section 2):
  * methods   -- SNI and TabCSDI, the two with a stopping rule (GAIN runs a
                 fixed 10000 iterations, MIWAE's loop is annotated "no early
                 stopping", so neither is affected);
  * datasets  -- NHANES (worst affected), Concrete (moderate), and one that was
                 never in trouble (AutoMPG), to confirm 200 epochs does not
                 over-train a healthy case;
  * budgets   -- 200 / 400 / 800, early stopping off throughout, 2 seeds.

    env PYTHONHASHSEED=2025 python tests/tabcsdi_knee.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "results" / "T2b_tabcsdi_budget"


def main() -> int:
    from tests.tabcsdi_epoch_probe import probe  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["NHANES", "Concrete", "AutoMPG"])
    ap.add_argument("--caps", type=int, nargs="*", default=[200, 400, 800])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--methods", nargs="*", default=["SNI", "TabCSDI"])
    ap.add_argument("--tables", default=str(CODE_ROOT / "data" / "derived_shuffled"))
    ap.add_argument("--masks", default=str(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"))
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before start (B48).",
              file=sys.stderr)
        return 2

    rows = []
    for ds in a.datasets:
        for cap in a.caps:
            for seed in a.seeds:
                for meth in a.methods:
                    # Early stopping disabled throughout: the question is what a
                    # given amount of training buys, and with patience on the cap
                    # is not the thing that decides it (finding B76).
                    r = probe(ds, seed, cap, Path(a.tables), Path(a.masks),
                              disable_early_stop=True, method=meth)
                    rows.append(r)
                    print(f"[{meth} {ds} forced {cap:<4} s{seed}] "
                          f"epochs_run={r['epochs_run']:<4} "
                          f"R2={r['cont_R2']:>10.4f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "knee_scan.csv", index=False)
    print("\n" + df[["method", "dataset", "epochs_run", "cont_R2"]]
          .to_string(index=False))

    print("\nbudget response (median R2 across seeds), per method and dataset:")
    piv = df.pivot_table(index=["method", "dataset"], columns="epochs_cap",
                         values="cont_R2", aggfunc="median")
    print(piv.round(4).to_string())
    print("\nhow much is still on the table at the configured 200 epochs:")
    for (meth, ds), row in piv.iterrows():
        if 200 in row and 800 in row and pd.notna(row[200]) and pd.notna(row[800]):
            print(f"  {meth:<8} {ds:<10} R2 200 -> 800 : "
                  f"{row[200]:+.4f} -> {row[800]:+.4f}  "
                  f"(delta {row[800] - row[200]:+.4f})")
    print(f"\nwrote {OUT / 'knee_scan.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
