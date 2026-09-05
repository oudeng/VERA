"""T2.1: render the before/after quantification the instruction requires.

"每处改动产出前后量化对照" -- every change gets a before/after number. The
builders already record what they did in their JSON reports; this turns those
into a table a reader can check, and refuses to render a dataset whose report is
missing rather than quietly omitting the row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"

REPORTS = {
    "MIMIC": DERIVED / "mimic_build_report.json",
    "eICU": DERIVED / "eicu_build_report.json",
    "NHANES": DERIVED / "nhanes_build_report.json",
    "CDC2022": DERIVED / "cdc_build_report.json",
}


def _rows_for(dataset: str, rep: dict) -> List[dict]:
    out: List[dict] = []
    src = rep.get("source_shape") or rep.get("merged_shape") or rep.get("shapes")
    fin = (rep.get("final") or {}).get("shape")
    if src and fin:
        out.append({"dataset": dataset, "change": "overall shape",
                    "before": f"{src}", "after": f"{fin}",
                    "detail": "rows x columns"})

    for s in rep.get("steps", []):
        step = s.get("step")
        cols = s.get("columns") or ([s["column"]] if s.get("column") else [])
        if step == "drop":
            out.append({"dataset": dataset, "change": f"drop: {s.get('group','')}",
                        "before": f"{len(cols)} columns present",
                        "after": "removed",
                        "detail": ", ".join(cols) + " -- " + str(s.get("reason", ""))})
        elif step == "reclassify_as_target":
            out.append({"dataset": dataset, "change": "reclassify as target",
                        "before": "imputable feature", "after": "downstream target",
                        "detail": ", ".join(cols) + " -- " + str(s.get("reason", ""))})
        elif step == "winsorise":
            out.append({"dataset": dataset, "change": f"winsorise {cols[0]}",
                        "before": f"max {s.get('max_before')}",
                        "after": f"cap {s.get('cap')}",
                        "detail": f"{s.get('rows_affected')} rows affected; "
                                  f"raw kept in {s.get('raw_values_preserved_in','-')}"})
        elif step == "record_cap_without_applying":
            out.append({"dataset": dataset, "change": f"cap recorded, not applied: {cols[0]}",
                        "before": f"range {s.get('observed_range')}",
                        "after": f"cap {s.get('cap')} available to the metric layer",
                        "detail": f"{s.get('rows_above_cap')} rows above cap "
                                  f"({100*float(s.get('share_above_cap',0)):.1f}%)"})
        elif step == "complete_case":
            out.append({"dataset": dataset, "change": "complete case",
                        "before": f"{s.get('rows_before')} rows",
                        "after": f"{s.get('rows_after')} rows",
                        "detail": f"retention {100*float(s.get('retention',0)):.1f}%"})
        elif step == "encode":
            out.append({"dataset": dataset, "change": f"encode {s.get('column')}",
                        "before": "string", "after": str(s.get("mapping")),
                        "detail": ""})

    cc = rep.get("complete_case")
    if cc:
        out.append({"dataset": dataset, "change": "complete case",
                    "before": f"{cc.get('rows_before')} rows",
                    "after": f"{cc.get('rows_after')} rows",
                    "detail": f"retention {100*float(cc.get('retention',0)):.1f}%"})

    for gate, g in (rep.get("gates") or {}).items():
        out.append({"dataset": dataset, "change": f"skip-aware gate: {gate}",
                    "before": f"{g.get('missing_naive')} missing (skip-blind)",
                    "after": f"{g.get('missing_skip_aware')} missing",
                    "detail": f"prevalence {g.get('prevalence')}; recovered via "
                              f"{g.get('rows_recovered_by_parent')}"})

    sub = rep.get("subsample")
    if sub:
        out.append({"dataset": dataset, "change": "stratified subsample",
                    "before": f"{rep.get('shapes',{}).get('no_nans')}",
                    "after": f"n={sub.get('n')}",
                    "detail": f"stratified on {sub.get('stratify_on')}, "
                              f"seed {sub.get('seed')}, balance {sub.get('balance')}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "T2.1_datalayer"))
    a = ap.parse_args()

    rows: List[dict] = []
    missing: List[str] = []
    for ds, p in REPORTS.items():
        if not p.exists():
            missing.append(f"{ds}: {p}")
            continue
        rows += _rows_for(ds, json.loads(p.read_text()))

    if missing:
        for m in missing:
            print(f"[MISSING REPORT] {m}")
        raise SystemExit(f"{len(missing)} build report(s) absent; run the builders")

    df = pd.DataFrame(rows)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "before_after.csv", index=False)

    lines = ["| dataset | change | before | after | detail |",
             "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.dataset} | {r.change} | {r.before} | {r.after} | "
                     f"{str(r.detail)[:150]} |")
    (outdir / "before_after.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {outdir}/before_after.{{csv,md}}  ({len(df)} changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
