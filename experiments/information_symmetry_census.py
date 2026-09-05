"""Information-symmetry census across every axis comparison.

Rules: docs/T61_information_symmetry_rules.md SS2 (the first author's
precondition 1, 2026-08-29).

`docs/T4F_presentation_rule.md` admitted an object to the recovery and
stability tables because "truth is the external synthetic DAG -- not
circular". That covers circularity. It does not cover whether the objects
being compared had access to the same information. This census asks that
question of every axis, and its output determines the recompute list.

Each object's oracle flag is not asserted here and taken on trust: the
declared source line is READ FROM THE FILE and matched against the oracle
pattern, so a claim that an object does not read withheld cells fails if its
own source says otherwise.

    env PYTHONHASHSEED=2025 python experiments/information_symmetry_census.py
    env PYTHONHASHSEED=2025 python experiments/information_symmetry_census.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "results" / "T6_symmetry" / "information_symmetry_census.json"

#: an expression whose value depends on cells withheld from the imputer:
#: the ground-truth frame indexed by the masked rows
ORACLE_PAT = re.compile(
    r"complete\w*\[[^\]]+\][^\n]*\[(miss|miss_idx)\]"
    r"|truth\s*=\s*[^\n]*complete\w*")
#: an expression that reads only what the imputer itself produced
NO_ORACLE_PAT = re.compile(
    r"X_final\[|numeric\[|completed\[|missing\w*|\bmask_df\b|feature_importances_"
    r"|shap_values|_compute_correlation_prior|compute_P\(")


def _line(file: str, lineno: int) -> str:
    p = CODE_ROOT / file
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    if not 1 <= lineno <= len(lines):
        return ""
    # the expression may wrap; take the declared line and the next two
    return " ".join(l.strip() for l in lines[lineno - 1:lineno + 2])


#: The declared map. `oracle` is the CLAIM; the census verifies it against the
#: source at (file, line). `role` distinguishes an object being compared from
#: the reference it is compared against.
AXES = [
 {"axis": "recovery",
  "reference": "the external synthetic generating graph G "
               "(results/T2.5_pilot ground_truth_G)",
  "scored_by": "AUROC / AUPRC / P@K / SHD of each object's matrix against G",
  "reported_in": "main Table 4, ESM Table 22, Fig. 3 recovery cells, "
                 "t_final.recovery",
  "objects": [
    {"name": "SNI-D", "role": "compared", "oracle": False,
     "source_of_values": "the host's attention matrix",
     "file": "experiments/pilot_r21.py", "line": 151},
    {"name": "SNI-D-retrained", "role": "compared", "oracle": False,
     "source_of_values": "the retrained host's attention matrix",
     "file": "experiments/t4f_perm_on_sni.py", "line": 152},
    {"name": "TAP (P-alone)", "role": "compared", "oracle": False,
     "source_of_values": "correlation prior on the initial completion of the "
                         "masked table",
     "file": "experiments/prior_attribution.py", "line": 237},
    {"name": "MissForest-importance", "role": "compared", "oracle": False,
     "source_of_values": "the fitted forests' feature_importances_",
     "file": "experiments/pilot_r21.py", "line": 268},
    {"name": "SHAP-on-MissForest", "role": "compared", "oracle": False,
     "source_of_values": "TreeSHAP on the fitted forests, over MissForest's "
                         "own completed table",
     "file": "experiments/pilot_r21.py", "line": 276},
    {"name": "Permutation-on-MissForest", "role": "compared", "oracle": False,
     "source_of_values": "permutation_importance scored against y = "
                         "numeric[target], MissForest's OWN completion",
     "file": "experiments/pilot_r21.py", "line": 281},
    {"name": "Permutation-on-SNI", "role": "compared", "oracle": True,
     "source_of_values": "_ablate, whose error signal is NRMSE against the "
                         "WITHHELD true values of the masked cells",
     "file": "experiments/t4f_perm_on_sni.py", "line": 106},
  ]},
 {"axis": "leakage",
  "reference": "the injected leakage proxies and the random-proxy calibration "
               "quantile",
  "scored_by": "per-object detection of the injected proxy against a "
               "calibrated threshold",
  "reported_in": "main Table 8, Fig. 2, Fig. 3 leakage cells, ESM Table 30, "
                 "t_final.leakage",
  "objects": [
    {"name": "SNI-D", "role": "compared", "oracle": False,
     "source_of_values": "the host's attention matrix on the injected table",
     "file": "experiments/t42_leakage.py", "line": 391},
    {"name": "TAP (P)", "role": "compared", "oracle": False,
     "source_of_values": "compute_P on the masked injected table",
     "file": "experiments/t42_leakage.py", "line": 377},
    {"name": "MissForest-importance", "role": "compared", "oracle": False,
     "source_of_values": "run_missforest_family on the masked injected table",
     "file": "experiments/t42_leakage.py", "line": 378},
    {"name": "SHAP-on-MissForest", "role": "compared", "oracle": False,
     "source_of_values": "run_missforest_family on the masked injected table",
     "file": "experiments/t42_leakage.py", "line": 378},
    {"name": "Permutation-on-MissForest", "role": "compared", "oracle": False,
     "source_of_values": "run_missforest_family on the masked injected table",
     "file": "experiments/t42_leakage.py", "line": 378},
    {"name": "Permutation-on-SNI", "role": "compared", "oracle": True,
     "source_of_values": "the T3.2 ablation algorithm on the injected table, "
                         "error signal against the WITHHELD true values",
     "file": "experiments/t42_leakage.py", "line": 413},
  ]},
 {"axis": "stability",
  "reference": "cross-seed agreement of each object's own matrix "
               "(no external reference)",
  "scored_by": "pairwise Spearman across seeds",
  "reported_in": "main Table 5, the host band that t43_verdict reads from "
                 "perm_on_sni_real_stability.csv, and through it the "
                 "no-prior mechanism verdict",
  "objects": [
    {"name": "SNI-D", "role": "compared", "oracle": False,
     "source_of_values": "the host's attention matrix",
     "file": "experiments/five_way_stability.py", "line": 60},
    {"name": "TAP (P)", "role": "compared", "oracle": False,
     "source_of_values": "compute_P on the masked table",
     "file": "experiments/five_way_stability.py", "line": 76},
    {"name": "MissForest family (3 readouts)", "role": "compared",
     "oracle": False,
     "source_of_values": "run_missforest_family on the masked table",
     "file": "experiments/five_way_stability.py", "line": 89},
    {"name": "Permutation-on-SNI", "role": "compared", "oracle": True,
     "source_of_values": "the real-table A matrices, produced by the "
                         "truth-reading ablation in faithfulness.py; the "
                         "cross-seed statistic measures the matrix's "
                         "STABILITY rather than its accuracy, so the "
                         "asymmetry bites less here -- but the matrix is "
                         "oracle-derived and its competitors' are not",
     "file": "experiments/faithfulness.py", "line": 159},
  ]},
 {"axis": "faithfulness",
  "reference": "the ablation matrix A -- host behavior measured AGAINST THE "
               "WITHHELD VALUES, prospectively specified as such in "
               "docs/T32_faithfulness_decision_rule.md under '## Ground truth'",
  "scored_by": "row-wise Spearman of each object's matrix against A",
  "reported_in": "main Tables 6 and 7, Fig. 3 faithfulness cells, "
                 "t_final.faithfulness",
  "objects": [
    {"name": "A (the reference itself)", "role": "reference", "oracle": True,
     "source_of_values": "NRMSE over f's masked cells against the complete "
                         "table -- privileged by design, and legitimate for a "
                         "reference; what must be disclosed is its SEMANTICS, "
                         "that it is host behavior measured with privileged "
                         "information rather than host behavior simpliciter",
     "file": "experiments/faithfulness.py", "line": 159},
    {"name": "SNI-D", "role": "compared", "oracle": False,
     "source_of_values": "the host's attention matrix",
     "file": "experiments/faithfulness.py", "line": 111},
    {"name": "TAP (P)", "role": "compared", "oracle": False,
     "source_of_values": "compute_P on the initial completion",
     "file": "experiments/prior_attribution.py", "line": 209},
    {"name": "MissForest family (3 readouts)", "role": "compared",
     "oracle": False,
     "source_of_values": "run_missforest_family on the masked table",
     "file": "experiments/five_way_stability.py", "line": 89},
    {"name": "Permutation-on-SNI", "role": "excluded-by-rule", "oracle": True,
     "source_of_values": "excluded from this table by "
                         "docs/T4F_presentation_rule.md because its agreement "
                         "with A is 1 by construction",
     "file": "experiments/faithfulness.py", "line": 159},
  ]},
 {"axis": "cost",
  "reference": "wall clock and peak memory; no error signal",
  "scored_by": "seconds and bytes",
  "reported_in": "main Table 9, ESM Table 20",
  "objects": [
    {"name": "all six objects", "role": "compared", "oracle": False,
     "source_of_values": "timings of the same construction each object needs; "
                         "no object's TIMING depends on a withheld value",
     "file": "experiments/cost_probe.py", "line": 97},
  ]},
]


def verify(axes=AXES) -> dict:
    out = {"rules": "docs/T61_information_symmetry_rules.md SS2", "axes": []}
    recompute: list = []
    for ax in axes:
        objs = []
        for o in ax["objects"]:
            src = _line(o["file"], o["line"])
            looks_oracle = bool(ORACLE_PAT.search(src))
            looks_clean = bool(NO_ORACLE_PAT.search(src)) and not looks_oracle
            claim = o["oracle"]
            # the claim must be supported by the source it names; an object
            # claimed clean whose own line reads the withheld cells is a
            # contradiction, and so is the reverse
            if claim:
                verified = looks_oracle
            else:
                verified = looks_clean or not looks_oracle
            objs.append({**o, "source_line": src[:200],
                         "source_says_oracle": looks_oracle,
                         "claim_verified": verified})
        compared = [o for o in objs if o["role"] == "compared"]
        priv = [o["name"] for o in compared if o["oracle"]]
        unpriv = [o["name"] for o in compared if not o["oracle"]]
        verdict = ("ASYMMETRIC" if priv and unpriv
                   else "SYMMETRIC" if compared else "N/A")
        rec = {**{k: v for k, v in ax.items() if k != "objects"},
               "objects": objs, "verdict": verdict,
               "privileged_objects": priv,
               "unprivileged_objects": unpriv,
               "unverified_claims": [o["name"] for o in objs
                                     if not o["claim_verified"]]}
        out["axes"].append(rec)
        if verdict == "ASYMMETRIC":
            for name in priv:
                recompute.append({"axis": ax["axis"], "object": name,
                                  "reported_in": ax["reported_in"]})
    out["recompute_list"] = recompute
    out["summary"] = {
        "n_axes": len(out["axes"]),
        "asymmetric": [a["axis"] for a in out["axes"]
                       if a["verdict"] == "ASYMMETRIC"],
        "symmetric": [a["axis"] for a in out["axes"]
                      if a["verdict"] == "SYMMETRIC"],
        "unverified_claims": [c for a in out["axes"]
                              for c in a["unverified_claims"]],
        "reference_semantics_to_disclose": [
            o["name"] for a in out["axes"] for o in a["objects"]
            if o["role"] == "reference" and o["oracle"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    # the oracle pattern must fire on the known oracle lines and not on the
    # known clean ones, or the census proves nothing
    check(bool(ORACLE_PAT.search(
        'truth = pd.to_numeric(complete[f], errors="coerce").to_numpy(float)[miss]')),
        "oracle pattern fires on the recovery ablation's target")
    check(bool(ORACLE_PAT.search(
        'truth = pd.to_numeric(complete_i[f], errors="coerce" ).to_numpy(float)[miss_idx]')),
        "oracle pattern fires on the leakage ablation's target")
    check(not ORACLE_PAT.search("y = numeric[target]"),
          "oracle pattern does not fire on MissForest's own completion")
    check(not ORACLE_PAT.search(
        "P = compute_P(cat, cont, s, missing[list(missing.columns)], None)"),
        "oracle pattern does not fire on TAP's input")

    r = verify()
    check(not r["summary"]["unverified_claims"],
          f"every oracle claim is supported by its own source line "
          f"({r['summary']['unverified_claims']})")
    check(r["summary"]["asymmetric"], "at least one axis is asymmetric -- if "
                                      "none were, this census would be "
                                      "reporting the opposite of what the "
                                      "second finding established")
    check(len(r["recompute_list"]) == sum(
        len(a["privileged_objects"]) for a in r["axes"]
        if a["verdict"] == "ASYMMETRIC"),
        "the recompute list is exactly the privileged objects of the "
        "asymmetric axes")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = verify()
    for a in r["axes"]:
        print(f"\n=== {a['axis']}: {a['verdict']} ===")
        print(f"  reference: {a['reference'][:90]}")
        for o in a["objects"]:
            flag = "ORACLE" if o["oracle"] else "clean "
            print(f"    [{flag}] {o['role']:<17} {o['name']:<30} "
                  f"{'verified' if o['claim_verified'] else 'CLAIM UNVERIFIED'}")
        if a["verdict"] == "ASYMMETRIC":
            print(f"  -> privileged: {a['privileged_objects']}")
    print(f"\nASYMMETRIC axes: {r['summary']['asymmetric']}")
    print(f"RECOMPUTE LIST ({len(r['recompute_list'])}):")
    for x in r["recompute_list"]:
        print(f"  {x['axis']:<14} {x['object']:<22} -> {x['reported_in'][:70]}")
    print(f"\nreference semantics to disclose: "
          f"{r['summary']['reference_semantics_to_disclose']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
