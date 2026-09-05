"""T1.3.2 read-only diagnostic: would the epsilon=1e-4 stopping rule ever fire?

The P1 instruction asks whether R0's convergence criterion would trigger, and in
which iteration, if ``cat_vars`` were passed to ``_max_imputed_delta`` (P0 finding
B4: ``imputer.py:350`` omits it, so the categorical branch at ``:692-695`` is dead
code). This script answers that from the artifacts R0 already wrote — every run
saved ``convergence_curve.csv`` with the per-iteration delta — so no model is
re-run and nothing is modified.

Result: the answer does not depend on ``cat_vars`` at all, and the criterion could
never fire. See the module docstring of the report section for the argument, which
this script measures.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
R0_RESULTS = CODE_ROOT.parent / "project_sni_R0" / "results_all"
OUT = CODE_ROOT / "results" / "T1.3_equivalence"
TOL = 1e-4
DS_RE = re.compile(r"_(MIMIC|NHANES|eICU|ComCri|AutoMPG|Concrete)_")


def collect() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(R0_RESULTS / "sni_*" / "*" / "convergence_curve.csv")):
        exp = os.path.basename(os.path.dirname(f))
        grp = Path(f).relative_to(R0_RESULTS).parts[0]
        m = DS_RE.search(exp)
        for _, r in pd.read_csv(f).iterrows():
            rows.append(dict(group=grp, exp=exp,
                             dataset=m.group(1) if m else "?",
                             iteration=int(r["iteration"]),
                             delta=float(r["delta"]),
                             cont_loss=float(r["cont_loss"]),
                             cat_loss=float(r["cat_loss"])))
    return pd.DataFrame(rows)


def main() -> int:
    df = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r0_convergence_deltas.csv", index=False)

    print(f"R0 runs scanned: {df.exp.nunique()}   iteration records: {len(df)}")
    print(f"tolerance epsilon = {TOL}\n")

    per_ds = df.groupby("dataset")["delta"].agg(["count", "min", "median", "max"])
    per_ds["min_over_tol"] = per_ds["min"] / TOL
    print("=== delta by dataset ===")
    print(per_ds.to_string(float_format=lambda v: f"{v:,.4g}"))

    n_below = int((df.delta < TOL).sum())
    print(f"\niterations with delta < epsilon : {n_below} / {len(df)}")
    print(f"smallest delta anywhere          : {df.delta.min():.6g} "
          f"({df.delta.min() / TOL:,.0f} x epsilon)")

    piv = df.pivot_table(index="exp", columns="iteration", values="delta")
    if {1, 2, 3}.issubset(piv.columns):
        mono = int(((piv[3] < piv[2]) & (piv[2] < piv[1])).sum())
        worse = int((piv[3] > piv[1]).sum())
        print(f"\nruns with monotonically decreasing delta : {mono} / {len(piv)} "
              f"({100*mono/len(piv):.1f}%)")
        print(f"runs where iteration 3 delta EXCEEDS iteration 1 : {worse} / {len(piv)} "
              f"({100*worse/len(piv):.1f}%)")

    print("""
=== Interpretation ===

_max_imputed_delta (imputer.py:684-703) returns, for continuous columns, the
MAXIMUM ABSOLUTE CHANGE in raw unnormalised units, and takes np.nanmax across all
imputable columns. It is then compared against epsilon = 1e-4 (imputer.py:369).

Two consequences:

1. The criterion is dimensionally incoherent. eICU's urine_output_min reaches
   220912, so a delta on the order of 1e4 is the normal scale there; NHANES
   triglycerides reach 2684. A raw-scale maximum change can only fall below 1e-4
   if the imputation is already numerically frozen to four decimal places in every
   cell, which never happens in three iterations.

2. Passing cat_vars would change nothing. The categorical branch (:692-695)
   returns a CHANGED FRACTION in [0, 1], but the return value is np.nanmax over
   ALL columns, and the continuous columns' raw-scale deltas (measured minimum 4.0,
   maximum 83170) dominate any value a categorical column could contribute. So the
   dead-code defect identified in P0 as B4 is real but incidental: fixing it alone
   would not make the criterion fire.

The stopping rule therefore never ran; G=3 was always exhausted, which is why all
410 runs carry warning="did_not_converge". Worse, delta is not even trending down
in most runs, so "not yet converged" is a fair description of the state at G=3.

This bears directly on ESM_1_SNI_HISC_v5_5.tex:537, which attributes runtime
differences to "the number of EM iterations before convergence" - a quantity that
is constant at 3 across the entire benchmark.

NOTHING WAS CHANGED. Whether to rescale the criterion (e.g. per-column
standardized delta), drop it in favor of a declared fixed G, or keep it and
report honestly, is a P3 decision.
""")
    print(f"detail -> {OUT / 'r0_convergence_deltas.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
