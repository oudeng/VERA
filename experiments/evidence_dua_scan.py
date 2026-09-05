"""The published aggregate artifacts, scanned on their own terms.

Adjudication, 2026-09-05 (SS1): publishing `evidence/` was ratified on one
condition -- that the five artifacts pass a DUA/identifier scan of their own,
recorded, because *the* justification for publishing them is "cell-level
aggregate, zero row-level", not "the tree-wide scan covered them".

Those are different claims. The tree-wide scan in build_public_repo asks
"does any published file leak a restricted table?"; it is a net cast over 224
files. This asks the narrower and stronger question of the five files the
manuscript actually points a reviewer at: is every row an aggregate, is every
regime synthetic, and does a row-level identifier appear anywhere at all --
as a column name, as a JSON key, or in free text?

A scan that only ever passes proves nothing, so the selftest plants each
violation it claims to detect and requires the scan to fail.

    PYTHONHASHSEED=2025 python experiments/evidence_dua_scan.py
    PYTHONHASHSEED=2025 python experiments/evidence_dua_scan.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
PUBLIC = ROOT / "VERA_GitHub"

#: The primary keys of the restricted derived tables. Their presence in a
#: published file -- anywhere, in any role -- is the thing the DUA forbids.
ROW_KEYS = (r"subject_id", r"hadm_id", r"stay_id", r"patientunitstay",
            r"icustay", r"\bmrn\b", r"\bicustay_id\b", r"patienthealthsystem")

#: Regimes are simulation designs. A regime naming a cohort would mean a cell
#: was computed on restricted rows and carried the cohort into the open.
SYNTHETIC_REGIMES = {"interaction_xor", "linear_gaussian", "nonlinear_mixed"}

FILES = ("t_final.json", "lambda_check.json", "t42_summary.json",
         "fair_same_host_recovery.json", "fair_same_host_recovery_cells.csv")


def _scan_text(name: str, text: str) -> list:
    return [f"{name}: row-level identifier /{k}/ appears in the file"
            for k in ROW_KEYS if re.search(k, text, re.I)]


def scan(base: Path = None) -> dict:
    base = (base or PUBLIC) / "evidence"
    problems, seen = [], {}
    for name in FILES:
        f = base / name
        if not f.exists():
            problems.append(f"{name}: not published")
            continue
        text = f.read_text(errors="replace")
        seen[name] = f.stat().st_size
        problems += _scan_text(name, text)
        if name.endswith(".csv"):
            rows = list(csv.DictReader(io.StringIO(text)))
            problems += [f"{name}: column {c!r} is a row-level identifier"
                         for c in (rows[0] if rows else {})
                         if any(re.search(k, c, re.I) for k in ROW_KEYS)]
            regimes = {r.get("regime", "") for r in rows}
            extra = regimes - SYNTHETIC_REGIMES
            if extra:
                problems.append(f"{name}: non-synthetic regime(s) {sorted(extra)}")
            #: every row must be an aggregate over more than one record
            for i, r in enumerate(rows):
                n = r.get("n_rows_common") or r.get("n_rows_scored")
                if n is not None and str(n).strip().isdigit() and int(n) <= 1:
                    problems.append(f"{name}: row {i} aggregates {n} record(s)")
        else:
            #: dataset names may appear as KEYS (which cohort a number is
            #: about); they may not carry a row payload underneath.
            obj = json.loads(text)
            problems += _cohort_keys(name, obj)
    return {"files": seen, "n_files": len(seen), "problems": problems,
            "total_bytes": sum(seen.values()), "pass": not problems}


def _cohort_keys(name: str, obj, path: str = "") -> list:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(re.search(r, str(k), re.I) for r in ROW_KEYS):
                out.append(f"{name}: identifier-shaped key {k!r} at {path}")
            out += _cohort_keys(name, v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _cohort_keys(name, v, f"{path}[{i}]")
    return out


def _selftest() -> int:
    import tempfile
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    r = scan()
    c(r["n_files"] == len(FILES), f"all {len(FILES)} artifacts published "
                                  f"({r['n_files']})")
    c(r["pass"], f"the delivered artifacts scan clean: {r['problems'][:3]}")
    c(r["total_bytes"] < 200_000, f"aggregate, not a table dump "
                                  f"({r['total_bytes']} B)")

    #: --- the scan must FAIL on each violation it claims to detect ------ #
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "evidence"
        d.mkdir()
        src = PUBLIC / "evidence"
        for n in FILES:
            (d / n).write_bytes((src / n).read_bytes())
        base = Path(td)

        p = d / "t_final.json"
        keep = p.read_text()
        p.write_text(json.dumps({"cohort": {"subject_id": [1, 2, 3]}}))
        c(not scan(base)["pass"], "an identifier-shaped JSON key is caught")
        p.write_text(keep)

        q = d / "fair_same_host_recovery_cells.csv"
        keepq = q.read_text()
        q.write_text(keepq.replace("interaction_xor", "mimic_icu_cohort", 1))
        c(not scan(base)["pass"], "a non-synthetic regime is caught")
        q.write_text(keepq.replace("regime,", "hadm_id,", 1))
        c(not scan(base)["pass"], "an identifier column is caught")
        q.write_text(keepq)
        c(scan(base)["pass"], "and the restored copy scans clean again")

        (d / "lambda_check.json").unlink()
        c(not scan(base)["pass"], "a missing artifact is caught")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = scan()
    print(json.dumps({k: v for k, v in r.items() if k != "problems"},
                     ensure_ascii=False))
    for p in r["problems"]:
        print(f"  [RED] {p}")
    print(("[OK] " if r["pass"] else "[RED] ")
          + f"{r['n_files']} published aggregate artifacts, "
            f"{r['total_bytes']} B: cell-level aggregate, synthetic regimes "
            f"only, zero row-level identifiers")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
