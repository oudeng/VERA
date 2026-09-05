"""The recovery readout under information symmetry (T6.1 addendum 2026-08-29c).

Leg (iii) of the SS5.1 mechanism argument says the same-host behavioral probe
recovers at 0.920, against the strongest externally hosted readout's 0.912.
That comparison was made with the probe's error signal scored against
the values withheld from the imputer, while every object it is compared with
works from the masked table or from a host's own completion. The paper
disclosed the asymmetry; this removes it.

Scoring is t4f_perm_on_sni.stage_score's algorithm VERBATIM -- the same
load_cell, the same measured_rows intersection, the same score() -- with one
substitution: Permutation-on-SNI is read from PERM_noOracle_* instead of the
archived PERM_*. Three variants are scored side by side so that retraining and
the oracle removal stay separable:

    archived            the matrices behind the number in the paper today
    refit_oracle        freshly retrained host, error signal against withheld
                        values -- isolates what retraining alone moved
    refit_no_oracle     freshly retrained host, error signal against the host's
                        own completed table -- the symmetric object

    env PYTHONHASHSEED=2025 python experiments/no_oracle_recovery_readout.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

PILOT = CODE_ROOT / "results" / "T2.5_pilot"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
OUT = CODE_ROOT / "results" / "T6_symmetry"

REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SEEDS = [2025, 2026, 2027, 2028, 2029]
#: the three objects hosted OUTSIDE the SNI model -- leg (iii)'s comparator set
EXTERNAL = ["MissForest-importance", "SHAP-on-MissForest",
            "Permutation-on-MissForest"]
VARIANTS = {"archived": T4F / "PERM_{tag}.csv",
            "refit_oracle": OUT / "PERM_oracle_{tag}.csv",
            "refit_no_oracle": OUT / "PERM_noOracle_{tag}.csv"}


def _contained(p: Path) -> Path:
    """T6.1: this recompute writes inside results/T6_symmetry/ and nowhere
    else, so no archived readout can be edited by it."""
    rp = Path(p).resolve()
    if OUT.resolve() not in rp.parents:
        raise RuntimeError(f"refusing to write outside {OUT}: {rp}")
    return rp


def _cell_matrices(regime: str, seed: int):
    """Everything stage_score loads for one cell, plus the three variants."""
    from pilot_r21 import load_cell
    from t4f_perm_on_sni import PILOT_METHODS
    complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
    cols = list(complete.columns)
    tag = f"{regime}_s{seed}"

    def _rd(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, index_col=0).reindex(
            index=cols, columns=cols).fillna(0.0)

    base = {m: _rd(PILOT / f"D_{regime}_s{seed}_{m}.csv")
            for m in PILOT_METHODS}
    base["P-alone"] = _rd(PRIOR / f"P_synth_{regime}_s{seed}.csv")
    variants = {k: _rd(Path(str(v).format(tag=tag)))
                for k, v in VARIANTS.items()}
    return base, variants, G, cols


def score_all() -> dict:
    """Per-variant scoring, each on its OWN common row set (the archived
    recipe), and a joint scoring on the intersection of all three."""
    from pilot_r21 import measured_rows, score

    own_rows, joint_rows = [], []
    for regime in REGIMES:
        for seed in SEEDS:
            base, variants, G, cols = _cell_matrices(regime, seed)

            # --- each variant scored exactly as stage_score would have ---
            for vname, P in variants.items():
                mats = dict(base)
                mats["Permutation-on-SNI"] = P
                common = np.ones(len(cols), dtype=bool)
                for M in mats.values():
                    common &= measured_rows(M)
                for name, M in mats.items():
                    sc = score(M, G, keep=common)
                    own_rows.append({"variant": vname, "regime": regime,
                                     "seed": seed, "method": name,
                                     "n_rows_common": int(common.sum()), **sc})

            # --- one row set for all three, so the head-to-head is paired ---
            mats = dict(base)
            common = np.ones(len(cols), dtype=bool)
            for M in list(base.values()) + list(variants.values()):
                common &= measured_rows(M)
            for name, M in base.items():
                joint_rows.append({"regime": regime, "seed": seed,
                                   "method": name,
                                   "n_rows_common": int(common.sum()),
                                   **score(M, G, keep=common)})
            for vname, P in variants.items():
                joint_rows.append({"regime": regime, "seed": seed,
                                   "method": f"Permutation-on-SNI[{vname}]",
                                   "n_rows_common": int(common.sum()),
                                   **score(P, G, keep=common)})
    return {"own": pd.DataFrame(own_rows), "joint": pd.DataFrame(joint_rows)}


def _strongest_external(df: pd.DataFrame, method_col: str = "method") -> tuple:
    m = df[df[method_col].isin(EXTERNAL)].groupby(method_col).auroc.mean()
    return str(m.idxmax()), float(m.max())


def verdict(tables: dict) -> dict:
    own, joint = tables["own"], tables["joint"]
    out = {"regimes": REGIMES, "seeds": SEEDS, "n_cells": len(REGIMES) * len(SEEDS)}

    # --- primary: each variant on its own common rows -------------------- #
    per_variant = {}
    for v in VARIANTS:
        d = own[own.variant == v]
        probe = float(d[d.method == "Permutation-on-SNI"].auroc.mean())
        ext_name, ext = _strongest_external(d)
        per_variant[v] = {
            "probe_mean_auroc": probe,
            "strongest_external": ext_name,
            "strongest_external_mean_auroc": ext,
            "probe_minus_external": probe - ext,
            "probe_at_least_external": bool(probe >= ext),
            "all_object_means": d.groupby("method").auroc.mean()
                                 .round(6).to_dict(),
        }
    out["per_variant_own_rows"] = per_variant

    # --- paired: all variants on one row set ----------------------------- #
    ext_name, ext = _strongest_external(joint)
    jm = joint.groupby("method").auroc.mean()
    out["joint_rows"] = {
        "strongest_external": ext_name,
        "strongest_external_mean_auroc": float(ext),
        "means": jm.round(6).to_dict(),
        "probe_minus_external": {
            v: float(jm[f"Permutation-on-SNI[{v}]"] - ext) for v in VARIANTS},
        "probe_at_least_external": {
            v: bool(jm[f"Permutation-on-SNI[{v}]"] >= ext) for v in VARIANTS},
    }

    # --- per-cell paired differences, the estimator the axis uses -------- #
    piv = joint.pivot_table(index=["regime", "seed"], columns="method",
                            values="auroc")
    col = f"Permutation-on-SNI[refit_no_oracle]"
    d = (piv[col] - piv[ext_name]).dropna()
    by_seed = {int(s): d.xs(s, level="seed").to_numpy(float)
               for s in SEEDS}
    from t51_cluster_stats import sign_flip_exact, seed_boot_ci_T
    T = float(np.mean([float(np.median(v)) for v in by_seed.values()]))
    out["paired_no_oracle_vs_strongest_external"] = {
        "comparator": ext_name,
        "T_mean_of_seed_medians": T,
        "ci95_T_seedboot": seed_boot_ci_T(by_seed),
        "exact_sign_enumeration": sign_flip_exact(by_seed),
        "per_regime_median": {r: float(np.median(d.xs(r, level="regime")))
                              for r in REGIMES},
        "cells_favouring_probe": int((d > 0).sum()),
        "cells_total": int(len(d)),
    }

    # --- the pre-adjudicated branch, applied by code --------------------- #
    stands = (per_variant["refit_no_oracle"]["probe_at_least_external"]
              and out["joint_rows"]["probe_at_least_external"]
              ["refit_no_oracle"])
    disagree = (per_variant["refit_no_oracle"]["probe_at_least_external"]
                != out["joint_rows"]["probe_at_least_external"]
                ["refit_no_oracle"])
    out["branch"] = {
        "rule": "docs/T61_information_symmetry_rules.md addendum 2026-08-29c",
        "own_rows_says": per_variant["refit_no_oracle"]
                         ["probe_at_least_external"],
        "joint_rows_says": out["joint_rows"]["probe_at_least_external"]
                           ["refit_no_oracle"],
        "the_two_disagree": bool(disagree),
        # the rule says the intersection governs if they disagree
        "leg_iii": ("STANDS" if (out["joint_rows"]["probe_at_least_external"]
                                 ["refit_no_oracle"])
                    else "WITHDRAWN_OR_NARROWED"),
        "note": ("the intersection version governs by rule when the two "
                 "disagree" if disagree else "both versions agree"),
    }
    if not stands and out["branch"]["leg_iii"] == "STANDS":
        out["branch"]["note"] += ("; own-row version disagrees and is "
                                  "reported beside it")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = score_all()
    tables["own"].to_csv(_contained(OUT / "no_oracle_recovery_cells_own.csv"),
                         index=False)
    tables["joint"].to_csv(_contained(OUT / "no_oracle_recovery_cells_joint.csv"),
                           index=False)
    v = verdict(tables)
    _contained(OUT / "no_oracle_recovery.json").write_text(
        json.dumps(v, indent=1))

    pv = v["per_variant_own_rows"]
    print("mean AUROC over the 15 cells, each variant on its own common rows")
    for name, r in pv.items():
        print(f"  {name:16s} probe {r['probe_mean_auroc']:.4f}   "
              f"strongest external ({r['strongest_external']}) "
              f"{r['strongest_external_mean_auroc']:.4f}   "
              f"delta {r['probe_minus_external']:+.4f}   "
              f"probe>=external {r['probe_at_least_external']}")
    j = v["joint_rows"]
    print(f"\non one row set for all variants (comparator "
          f"{j['strongest_external']} = "
          f"{j['strongest_external_mean_auroc']:.4f}):")
    for k, d in j["probe_minus_external"].items():
        print(f"  {k:16s} delta {d:+.4f}   "
              f"probe>=external {j['probe_at_least_external'][k]}")
    p = v["paired_no_oracle_vs_strongest_external"]
    print(f"\npaired (no-oracle vs {p['comparator']}): "
          f"T={p['T_mean_of_seed_medians']:+.4f}  "
          f"CI {p['ci95_T_seedboot']}  "
          f"exact p={p['exact_sign_enumeration'].get('p_two_sided')}  "
          f"cells favoring probe {p['cells_favouring_probe']}/"
          f"{p['cells_total']}")
    print(f"\nLEG (iii): {v['branch']['leg_iii']}  ({v['branch']['note']})")
    print(f"\nwrote {OUT / 'no_oracle_recovery.json'}")
    return 0


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    for v, tmpl in VARIANTS.items():
        miss = [f"{r}_s{s}" for r in REGIMES for s in SEEDS
                if not Path(str(tmpl).format(tag=f"{r}_s{s}")).exists()]
        chk(not miss, f"all 15 matrices present for '{v}' ({miss[:3]})")

    # the scorer and the row-selection are the archived ones, not copies
    import t4f_perm_on_sni as t4f
    import pilot_r21
    chk(t4f.PILOT_METHODS[0] == "SNI-D" and len(t4f.PILOT_METHODS) == 4,
        f"PILOT_METHODS taken from t4f ({t4f.PILOT_METHODS})")
    chk(callable(pilot_r21.score) and callable(pilot_r21.measured_rows),
        "score() and measured_rows() imported from the pilot, not reimplemented")

    # the write guard actually refuses
    try:
        _contained(CODE_ROOT / "results" / "T4_perm_on_sni" / "x.csv")
        chk(False, "the write guard refuses a path outside T6_symmetry")
    except RuntimeError:
        chk(True, "the write guard refuses a path outside T6_symmetry")

    # the archived variant must reproduce the archived cells it came from
    arch = pd.read_csv(T4F / "t4f_sixway_cells.csv")
    a = arch[arch.method == "Permutation-on-SNI"].auroc.mean()
    chk(abs(a - 0.920) < 0.01,
        f"the archived Permutation-on-SNI mean AUROC is the paper's 0.920 "
        f"({a:.4f})")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
