"""T5.3 correction record: the superseded readouts beside the corrected ones.

Rules: docs/T53_input_correction_rules.md SS3 -- "a correction that is not
shown beside what it replaced is not auditable". This reads both artifacts
and emits the difference per variant per table; it computes no new science.

    env PYTHONHASHSEED=2025 python experiments/t53_correction_record.py
    env PYTHONHASHSEED=2025 python experiments/t53_correction_record.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

FAM = CODE_ROOT / "results" / "T5_family"
OLD = FAM / "superseded_20260829_wrong_input"
OUT = CODE_ROOT / "results" / "T6_lineage" / "t53_correction_record.json"
#: the two variants the correction touched; the others were already correct
CORRECTED = ["abs_spearman", "mutual_information"]
UNTOUCHED = ["observed_only_tap", "uniform", "random"]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _stat(block: dict, key: str):
    v = block.get(key)
    return v if isinstance(v, dict) else None


def build(new_path: Path = FAM / "tapfam_summary.json",
          old_path: Path = OLD / "tapfam_summary.json",
          out_path: Path = OUT) -> dict:
    new = json.loads(new_path.read_text())
    old = json.loads(old_path.read_text())
    rec = {
        "rules": "docs/T53_input_correction_rules.md",
        "what_changed": "the table the abs_spearman and mutual_information "
                        "variants are computed on: the pre-mask ground-truth "
                        "table before, the initial completion of the masked "
                        "table after -- the same table the archived TAP_0 "
                        "used. Nothing else changed: no variant definition, "
                        "no seed, no discretisation, no threshold.",
        "verification": "results/T6_lineage/data_lineage_audit.json -- the "
                        "corrected inputs classify CLEAN-COMPLETION on every "
                        "masked coordinate and their masked-coordinate digest "
                        "equals the archived TAP_0 input's",
        "superseded_artifacts": {
            "tapfam_summary.json": {"path": str(old_path.relative_to(CODE_ROOT)),
                                    "md5": _md5(old_path)},
            "tapfam_cells.csv": {
                "path": str((OLD / "tapfam_cells.csv").relative_to(CODE_ROOT)),
                "md5": _md5(OLD / "tapfam_cells.csv")},
        },
        "corrected_artifacts": {
            "tapfam_summary.json": {"path": str(new_path.relative_to(CODE_ROOT)),
                                    "md5": _md5(new_path)},
            "tapfam_cells.csv": {
                "path": str((FAM / "tapfam_cells.csv").relative_to(CODE_ROOT)),
                "md5": _md5(FAM / "tapfam_cells.csv")},
        },
        "datasets": {},
    }
    directions = []
    for ds in sorted(new.get("datasets", {})):
        nb, ob = new["datasets"][ds], old["datasets"].get(ds, {})
        per = {}
        for v in CORRECTED:
            n, o = _stat(nb, v), _stat(ob, v)
            if not (n and o):
                continue
            dT = round(float(n["T"]) - float(o["T"]), 6)
            per[v] = {
                "superseded": {"T": o["T"], "ci95_T": o.get("ci95_T"),
                               "p_exact": o.get("p_exact"),
                               "n_seeds": o.get("n_seeds")},
                "corrected": {"T": n["T"], "ci95_T": n.get("ci95_T"),
                              "p_exact": n.get("p_exact"),
                              "n_seeds": n.get("n_seeds")},
                "delta_T_corrected_minus_superseded": dT,
                "sign_changed": bool((float(o["T"]) < 0) != (float(n["T"]) < 0)),
            }
            directions.append(dT)
        for v in UNTOUCHED:
            n, o = _stat(nb, v), _stat(ob, v)
            if n and o:
                per[v] = {"unchanged_by_construction": True,
                          "identical": bool(json.dumps(n, sort_keys=True)
                                            == json.dumps(o, sort_keys=True))}
        rec["datasets"][ds] = per
    rec["direction_summary"] = {
        "n_corrected_cells": len(directions),
        "n_moved_up": sum(1 for d in directions if d > 0),
        "n_moved_down": sum(1 for d in directions if d < 0),
        "statement": "The contaminated inputs gave these two variants "
                     "information the comparison baseline did not have. The "
                     "REALISED effect of removing it was not one-directional: "
                     f"{sum(1 for d in directions if d > 0)} of "
                     f"{len(directions)} corrected readouts moved up and "
                     f"{sum(1 for d in directions if d < 0)} moved down. The "
                     "asymmetry was in the information, not in a uniform "
                     "advantage in the outcome, and both directions are "
                     "reported.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    return rec


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    r = build()
    check(set(r["datasets"]) == {"MIMIC", "eICU"}, "both tables recorded")
    for ds in r["datasets"]:
        for v in CORRECTED:
            e = r["datasets"][ds][v]
            check("superseded" in e and "corrected" in e,
                  f"{ds}/{v}: both readouts kept side by side")
            check(e["delta_T_corrected_minus_superseded"] ==
                  round(float(e["corrected"]["T"])
                        - float(e["superseded"]["T"]), 6),
                  f"{ds}/{v}: delta is the arithmetic difference")
    check(r["direction_summary"]["n_corrected_cells"] == 4,
          "four corrected readouts (2 variants x 2 tables)")
    check(r["superseded_artifacts"]["tapfam_summary.json"]["md5"]
          != r["corrected_artifacts"]["tapfam_summary.json"]["md5"],
          "the superseded artifact is a different file, not overwritten")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = build()
    for ds, per in r["datasets"].items():
        print(f"\n=== {ds} ===")
        for v in CORRECTED:
            e = per[v]
            print(f"  {v:<22} T {e['superseded']['T']:+.6f} -> "
                  f"{e['corrected']['T']:+.6f}  "
                  f"(delta {e['delta_T_corrected_minus_superseded']:+.6f}"
                  f"{', SIGN CHANGED' if e['sign_changed'] else ''})")
            print(f"  {'':<22} p {e['superseded']['p_exact']:.4f} -> "
                  f"{e['corrected']['p_exact']:.4f}")
        for v in UNTOUCHED:
            if v in per:
                print(f"  {v:<22} {'identical' if per[v].get('identical') else 'CHANGED -- investigate'}")
    print("\n" + r["direction_summary"]["statement"])
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
