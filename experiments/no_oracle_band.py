"""The no-oracle host band, and the T4.3 chain it feeds.

Rules: docs/T61_information_symmetry_rules.md, addendum 2026-08-29b, committed
before this file existed.

Two of the four legs of the manuscript's mechanism argument rest on
`Permutation-on-SNI`: (i) over-stability, where the HOST BAND *is* that
object's cross-seed stability, and (iii) the recovery window. This computes
the band from the no-oracle matrices by `t4f_perm_on_sni.stage_real`'s
algorithm verbatim, then re-evaluates the T4.3 O1 gate and the mechanism
ladder that take the band as an input.

It changes nothing in place: every output goes to results/T6_symmetry/.

    env PYTHONHASHSEED=2025 python experiments/no_oracle_band.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
SYM = CODE_ROOT / "results" / "T6_symmetry"
OUT = SYM / "no_oracle_band.json"

DATASETS = ["MIMIC", "eICU"]
SEEDS = [1, 2, 3, 5, 8]


def _band(mats: dict) -> dict:
    """t4f_perm_on_sni.stage_real's algorithm, verbatim: pairwise Spearman
    over the entries selected by the FIRST seed's notna mask, all ten pairs."""
    from scipy.stats import spearmanr
    seeds = sorted(mats)
    first = mats[seeds[0]]
    sel = first.notna().to_numpy()
    rows = []
    for a, b in combinations(seeds, 2):
        A = mats[a].to_numpy(float)[sel]
        B = mats[b].to_numpy(float)[sel]
        ja = []
        for f in first.index:
            ra = mats[a].loc[f].dropna()
            rb = mats[b].loc[f].dropna()
            ta = set(ra.sort_values(ascending=False).index[:3])
            tb = set(rb.sort_values(ascending=False).index[:3])
            ja.append(len(ta & tb) / len(ta | tb))
        rows.append({"a": a, "b": b,
                     "spearman": round(float(spearmanr(A, B).statistic), 4),
                     "top3_jaccard": round(float(np.mean(ja)), 4)})
    sp = [r["spearman"] for r in rows]
    return {"pairs": rows, "n_pairs": len(rows),
            "mean": round(float(np.mean(sp)), 4),
            "min": round(float(np.min(sp)), 4),
            "max": round(float(np.max(sp)), 4),
            "top3_jaccard_mean": round(
                float(np.mean([r["top3_jaccard"] for r in rows])), 4)}


def _verdicts(ds: str, host_or: float, host_no: float, d_prior: float) -> dict:
    """Re-evaluate the T4.3 O1 gate and the mechanism ladder with each band.

    O1, O2 and the O3 tier are read from the shipped verdict artifact -- they
    do not depend on the band. Only T_stab does, so only the O1 comparison can
    move. The ladder function is imported, not reimplemented.
    """
    from t43_verdict import mechanism
    shipped_path = CODE_ROOT / "results" / "T4_noprior" / "t43_verdict.json"
    if not shipped_path.exists():
        return {"verdict_source": "NOT FOUND -- reported, not guessed"}
    v = json.loads(shipped_path.read_text())
    # T4.3's committed scope is the eICU control; MIMIC is untested for the
    # NoPrior variant, so the ladder is re-evaluated only where it was run.
    if v["scope"]["table"] != ds:
        return {"verdict_source": str(shipped_path.relative_to(CODE_ROOT)),
                "not_in_scope": f"T4.3's committed scope is "
                                f"{v['scope']['table']} only; the mechanism "
                                f"ladder was never run on {ds} and is not "
                                f"re-evaluated here"}
    obs = v["observables"]
    o1 = float(obs["O1_stability_own"])
    o2 = float(obs["O2_median_rho"])
    tier = obs["O3_parent_tier"]
    out = {"verdict_source": str(shipped_path.relative_to(CODE_ROOT)),
           "O1": round(o1, 4), "O2": round(o2, 4), "O3_tier": tier,
           "shipped_T_stab": v["thresholds"]["T_stab"],
           "shipped_host_band": v["thresholds"]["host_band_withprior"],
           "shipped_category": v["mechanism_verdict"],
           "shipped_deviations": v["mechanism_deviations"]}
    for tag, host in (("oracle", host_or), ("noOracle", host_no)):
        t_stab = host + 0.5 * (d_prior - host)
        cat, dev = mechanism(tier, o1, o2, t_stab)
        out[f"verdict_with_{tag}_band"] = {
            "T_stab": round(t_stab, 4), "category": cat, "deviations": dev}
    out["category_changed"] = (
        out["verdict_with_oracle_band"]["category"]
        != out["verdict_with_noOracle_band"]["category"])
    out["category_changed_vs_shipped"] = (
        out["shipped_category"]
        != out["verdict_with_noOracle_band"]["category"])
    return out


def _load(kind: str, ds: str) -> dict:
    if kind == "oracle":
        return {s: pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv",
                               index_col=0) for s in SEEDS}
    return {s: pd.read_csv(SYM / f"A_noOracle_{ds}_seed{s}_cpu_t2.csv",
                           index_col=0) for s in SEEDS}


def _tap_agreement(ds: str) -> dict:
    """Spearman agreement with TAP for the information-symmetric readout.

    The archived column (results/T4_perm_on_sni/perm_on_sni_agreement_with_P.csv)
    was computed from the oracle-signal ablation matrices. This is the SAME
    arithmetic -- t4f_perm_on_sni's agreement loop, verbatim: the first-seed
    notna mask per target row, flattened, Spearman against the TAP matrix --
    applied to the no-oracle matrices instead. Nothing is retrained.
    """
    from scipy.stats import spearmanr
    P = pd.read_csv(CODE_ROOT / "results" / "T2g_prior_attribution"
                    / f"P_{ds}_seed1_cpu_t2.csv", index_col=0)
    per = {}
    for s_ in SEEDS:
        A = pd.read_csv(SYM / f"A_noOracle_{ds}_seed{s_}_cpu_t2.csv",
                        index_col=0)
        sel = A.notna()
        av, pv = [], []
        for f in A.index:
            cols = list(A.columns[sel.loc[f]])
            av.extend(A.loc[f, cols].to_numpy(float))
            pv.extend(P.loc[f, cols].to_numpy(float))
        per[str(s_)] = round(float(spearmanr(av, pv).statistic), 4)
    return {"per_seed": per,
            "mean": round(float(np.mean(list(per.values()))), 4),
            "algorithm": "t4f_perm_on_sni agreement loop, verbatim, on the "
                         "no-oracle ablation matrices",
            "inputs": f"results/T6_symmetry/A_noOracle_{ds}_seed*_cpu_t2.csv "
                      f"+ results/T2g_prior_attribution/P_{ds}_seed1_cpu_t2.csv",
            "retraining": "none -- pure readout of existing matrices"}


def run() -> dict:
    rec = {"rules": "docs/T61_information_symmetry_rules.md addendum 2026-08-29b",
           "algorithm": "t4f_perm_on_sni.stage_real verbatim (pairwise "
                        "Spearman over the first seed's notna mask, 10 pairs)",
           "datasets": {}}

    # the archived band, as shipped, for the control
    shipped = pd.read_csv(T4F / "perm_on_sni_real_stability.csv")

    for ds in DATASETS:
        b_or = _band(_load("oracle", ds))
        b_no = _band(_load("noOracle", ds))
        ship_mean = float(shipped[shipped.dataset == ds].spearman.mean())
        fw = json.loads((FIVEWAY / "fiveway_summary.json").read_text())
        d_prior = float(fw[ds]["SNI-D"]["stability_mean"])

        def gate(host):
            t_stab = host + 0.5 * (d_prior - host)
            return {"host_band": round(host, 4),
                    "D_prior_stability": round(d_prior, 4),
                    "T_stab": round(t_stab, 4)}

        rec["datasets"][ds] = {
            "control_recomputed_oracle_band_vs_shipped": {
                "recomputed_mean": b_or["mean"], "shipped_mean": round(ship_mean, 4),
                "abs_diff": round(abs(b_or["mean"] - ship_mean), 6)},
            "band_oracle": b_or,
            "band_noOracle": b_no,
            "delta_mean_noOracle_minus_oracle":
                round(b_no["mean"] - b_or["mean"], 4),
            "D_prior_stability": round(d_prior, 4),
            "gate_with_oracle_band": gate(b_or["mean"]),
            "gate_with_noOracle_band": gate(b_no["mean"]),
            "leg_i_still_holds": bool(b_no["mean"] < d_prior),
            "leg_i_margin": round(d_prior - b_no["mean"], 4),
            "leg_i_margin_with_oracle_band": round(d_prior - b_or["mean"], 4),
            "bands_disjoint_from_D": bool(b_no["max"] < d_prior),
            # Eighth review P0-3: the symmetric row of Table 5 needs the same
            # last column the archived row has, computed on the SAME caliber
            # as the rest of its own row. It is a pure readout of matrices
            # that already exist -- no retraining, seconds -- so there is no
            # excuse for printing the archived caliber's value beside a
            # symmetric band, and none for leaving the cell empty either.
            "agreement_with_TAP_noOracle": _tap_agreement(ds),
            **_verdicts(ds, b_or["mean"], b_no["mean"], d_prior),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    return rec


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    r = run()
    for ds in DATASETS:
        d = r["datasets"][ds]
        ctl = d["control_recomputed_oracle_band_vs_shipped"]
        check(ctl["abs_diff"] < 5e-4,
              f"{ds}: CONTROL -- recomputing the band from the archived "
              f"matrices reproduces the shipped band "
              f"({ctl['recomputed_mean']} vs {ctl['shipped_mean']})")
        check(d["band_noOracle"]["n_pairs"] == 10,
              f"{ds}: ten seed pairs, as shipped")
        # T_stab is stored rounded to 4dp; compare at that precision
        want = round(d["band_oracle"]["mean"] + 0.5
                     * (d["D_prior_stability"] - d["band_oracle"]["mean"]), 4)
        check(abs(d["gate_with_oracle_band"]["T_stab"] - want) < 1e-4,
              f"{ds}: T_stab is host + 0.5*(D_prior - host), verbatim")
        if "not_in_scope" in d:
            check(True, f"{ds}: {d['not_in_scope'][:70]}")
        else:
            check(d["verdict_with_oracle_band"]["category"]
                  == d["shipped_category"],
                  f"{ds}: CONTROL -- re-evaluating the ladder with the ORACLE "
                  f"band reproduces the shipped verdict "
                  f"({d['verdict_with_oracle_band']['category']})")
            check(d["verdict_with_noOracle_band"]["category"] in
                  ("MECH-PRIOR", "MECH-OPTIMIZED", "MECH-MIXED"),
                  f"{ds}: the ladder returns a declared category")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = run()
    for ds in DATASETS:
        d = r["datasets"][ds]
        print(f"\n=== {ds} ===")
        ctl = d["control_recomputed_oracle_band_vs_shipped"]
        print(f"  CONTROL   recomputed oracle band {ctl['recomputed_mean']} "
              f"vs shipped {ctl['shipped_mean']}  (|diff| {ctl['abs_diff']})")
        print(f"  band  oracle    mean {d['band_oracle']['mean']}  "
              f"[{d['band_oracle']['min']}, {d['band_oracle']['max']}]")
        print(f"  band  no-oracle mean {d['band_noOracle']['mean']}  "
              f"[{d['band_noOracle']['min']}, {d['band_noOracle']['max']}]  "
              f"(delta {d['delta_mean_noOracle_minus_oracle']:+})")
        print(f"  D (with prior) cross-seed stability: "
              f"{d['D_prior_stability']}")
        print(f"  leg (i) over-stability holds: {d['leg_i_still_holds']}  "
              f"margin {d['leg_i_margin']:+}")
        print(f"  T_stab  with oracle band {d['gate_with_oracle_band']['T_stab']}"
              f"   with no-oracle band "
              f"{d['gate_with_noOracle_band']['T_stab']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
