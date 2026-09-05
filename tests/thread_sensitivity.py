"""Does SNI's CPU result depend on the BLAS thread count?

Raised by an accident during the T2.0 gate. A pilot run of MIMIC seed 1 and the
full gate's run of the same dataset, seed, codebase and PYTHONHASHSEED (the
`hash_probe` was 3944 in both) produced different numbers:

    Accuracy   0.307482587724  vs  0.310849145115   (3.4e-3)
    cont_MB    0.318261400242  vs  0.293648230801   (2.5e-2)

The only known difference between them is the thread count. If that is the
cause, then a seed alone does not determine an SNI run on CPU and the
reproducibility statement has to say so -- the same class of problem as B48,
where a recorded seed also failed to determine the run, but arising from
floating-point non-associativity in parallel reductions rather than from hash
randomization.

The gate itself is unaffected: both sides of it run at a single fixed thread
count and agree bit for bit there. What is at stake is whether a third party can
reproduce our numbers on their own machine.

Run AFTER the gates, since it needs the CPU to itself to be meaningful:

    env PYTHONHASHSEED=2025 python tests/thread_sensitivity.py --side r0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "T2.0_gate" / "thread_sensitivity"

#: One process at a time, so the counts mean what they say.
THREAD_COUNTS = [1, 8, 24, 32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="r0", choices=["r0", "port"])
    ap.add_argument("--dataset", default="MIMIC")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, nargs="*", default=THREAD_COUNTS)
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts. Otherwise this measures B48, not threading.",
              file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.threads:
        outroot = OUT / f"t{n}"
        env = dict(os.environ, SNI_EQ_THREADS=str(n))
        cmd = [sys.executable, str(ROOT / "tests" / "gate_equivalence.py"),
               "--side", a.side, "--device", "cpu",
               "--datasets", a.dataset, "--seeds", str(a.seed),
               "--outroot", str(outroot)]
        print(f"[run] threads={n} ...", flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            # Listed, never silently skipped.
            print(f"[FAIL] threads={n} rc={r.returncode}\n{r.stderr[-800:]}")
            rows.append({"threads": n, "status": f"FAILED rc={r.returncode}"})
            continue
        f = (outroot / f"cpu_{a.side}" / f"{a.dataset}_s{a.seed}"
             / "metrics_summary.json")
        m = json.loads(f.read_text())
        rows.append({"threads": n, "status": "ok",
                     **{k: v for k, v in m.items()
                        if isinstance(v, (int, float))
                        and k not in ("runtime_sec", "runtime_sec_wall")}})
        # Per iteration, not at the end (B79, L2). Each of these is a full SNI
        # training run; losing four of them to a crash in the fifth is avoidable.
        pd.DataFrame(rows).to_csv(
            OUT / f"thread_sensitivity_{a.side}_{a.dataset}_s{a.seed}.csv",
            index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"thread_sensitivity_{a.side}_{a.dataset}_s{a.seed}.csv",
              index=False)
    print("\n" + df.to_string(index=False))

    ok = df[df.status == "ok"]
    if len(ok) > 1:
        print("\nspread across thread counts (0 means thread-independent):")
        for c in ok.columns:
            if c in ("threads", "status") or not pd.api.types.is_numeric_dtype(ok[c]):
                continue
            spread = float(ok[c].max() - ok[c].min())
            flag = "  <== varies" if spread > 1e-12 else ""
            print(f"  {c:<24} {spread:.6e}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
