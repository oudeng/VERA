"""Re-apply the current plausibility criterion to smoke results already on disk.

The smoke workers load `plausibility` at import, so a run in flight keeps the
criterion it started with. That is correct -- a running experiment should not
change its own rules mid-flight -- but it means the `status` column in
`smoke_*.csv` can be stale once the criterion is corrected.

Both corrections that happened during T2b.3 are the reason this exists:

* the absolute floor of R^2 > -1 flagged GAIN on tables where R0 records the
  same behavior, i.e. it tested the method's quality rather than our harness;
* the "unchanged since R0" set wrongly included eICU and Concrete, so a run was
  flagged for scoring *above* R0's band on a table R0 never had.

Re-scoring is not the same as loosening: this prints what changed and what still
fails, so a criterion that quietly turned into a rubber stamp would be visible.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"\[(\w+)\s*\]\s+(\S+)/(\S+)\s+R2=\s*(\S+)\s+NRMSE=(\S+)\s+([\d.]+)s")


def from_logs(pattern: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            g = LINE.match(line)
            if g:
                rows.append(dict(status_at_runtime=g.group(1), method=g.group(2),
                                 dataset=g.group(3), cont_R2=float(g.group(4)),
                                 cont_NRMSE=float(g.group(5)),
                                 runtime_sec=float(g.group(6)), source=Path(f).name))
    return pd.DataFrame(rows)


def main() -> int:
    import importlib.util
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(CODE_ROOT / "results" / "T2b_smoke" / "*.log"))
    ap.add_argument("--out", default=str(CODE_ROOT / "results" / "T2b_smoke" / "rescored.csv"))
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "smk", CODE_ROOT / "tests" / "smoke_nine_methods.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["smk"] = m
    spec.loader.exec_module(m)
    ref = m.r0_reference()

    df = from_logs(a.logs)
    if df.empty:
        print(f"no smoke results under {a.logs}")
        return 1
    df["problems"] = [
        "; ".join(m.plausibility(r.method, r.dataset, r.cont_R2, r.cont_NRMSE, ref))
        for r in df.itertuples()]
    df["status"] = ["ok" if not p else "IMPLAUSIBLE" for p in df.problems]
    df.to_csv(a.out, index=False)

    print(f"{len(df)} runs, {df.dataset.nunique()} datasets, "
          f"{df.method.nunique()} methods")
    print(f"  at runtime : {df.status_at_runtime.value_counts().to_dict()}")
    print(f"  re-scored  : {df.status.value_counts().to_dict()}")
    print(f"  R0 reference bands used for: "
          f"{sorted({ds for (_, ds) in ref} - m.REBUILT)}")

    changed = df[df.status_at_runtime.str.lower() != df.status.str.lower()]
    if len(changed):
        print(f"\n{len(changed)} verdict(s) changed:")
        print(changed[["method", "dataset", "cont_R2",
                       "status_at_runtime", "status"]].to_string(index=False))
    still = df[df.status != "ok"]
    print(f"\nstill failing: "
          f"{still[['method','dataset','cont_R2','problems']].to_string(index=False)}"
          if len(still) else "\nstill failing: none")
    print(f"\nwrote {a.out}")
    return 0 if still.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
