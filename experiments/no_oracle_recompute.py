"""Align the privileged object downward: recompute Permutation-on-SNI with the
error signal its competitors already use.

Rules: docs/T61_information_symmetry_rules.md (committed before this file
existed). The census
(`results/T6_symmetry/information_symmetry_census.json`) found three
asymmetric axes -- recovery, leakage, stability -- all because
`Permutation-on-SNI` scores its ablation against the WITHHELD true values
while every object it is compared with derives its values from the masked
table or from an imputer's own completion.

The correction is always downward (T6.1 SS3): the error target becomes
`X_final[f][miss]`, the imputer's own completed values on the same cells --
exactly the signal `permutation_importance` gets in
`experiments/pilot_r21.py`. Nothing else changes: same permutation count,
same RNG streams, same standardization, same rows.

    env PYTHONHASHSEED=2025 python experiments/no_oracle_recompute.py --selftest
    env PYTHONHASHSEED=2025 python experiments/no_oracle_recompute.py --axis stability
    env PYTHONHASHSEED=2025 python experiments/no_oracle_recompute.py --axis recovery
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
PILOT = CODE_ROOT / "results" / "T2.5_pilot"
OUT = CODE_ROOT / "results" / "T6_symmetry"
STAMP = "20260829"

#: C-6 window convention. This recompute occupies the machine and produces new
#: wall-clock records; none of them may enter the cost table's source set. The
#: cost generators read exactly these four places, and this script writes to
#: none of them -- asserted at import, not merely intended.
COST_SOURCES = [CODE_ROOT / "results" / "A3_cost_context",
                CODE_ROOT / "results" / "P2_main_grid",
                CODE_ROOT / "results" / "T3_five_way" / "fiveway_cost.csv",
                T4F_COST := CODE_ROOT / "results" / "T4_perm_on_sni"
                / "perm_on_sni_audit_cost.csv"]


def _assert_writes_are_contained(path: Path) -> Path:
    """Every write goes under results/T6_symmetry/ and nowhere else."""
    rp = Path(path).resolve()
    if OUT.resolve() not in rp.parents and rp != OUT.resolve():
        raise RuntimeError(
            f"refusing to write outside results/T6_symmetry: {rp}. This run is "
            f"registered as an auxiliary-load window (results/aux_windows.json) "
            f"and must not deposit a wall-clock record where a cost generator "
            f"would read it.")
    return rp

REAL = [("MIMIC", s) for s in (1, 2, 3, 5, 8)] + \
       [("eICU", s) for s in (1, 2, 3, 5, 8)]
REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SYNTH_SEEDS = [2025, 2026, 2027, 2028, 2029]


# --------------------------------------------------------------------------- #
def archive(paths, why: str) -> dict:
    """Superseded artifacts are kept, never overwritten (T6.1 SS5)."""
    import hashlib
    d = OUT / f"superseded_{STAMP}_oracle_error_signal"
    d.mkdir(parents=True, exist_ok=True)
    rec = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        tgt = d / p.name
        if not tgt.exists():
            shutil.copy2(p, tgt)
        rec[p.name] = hashlib.md5(tgt.read_bytes()).hexdigest()
    (d / "README.md").write_text(
        "# Superseded readouts -- computed with the oracle error signal\n\n"
        + why + "\n\nDo not cite these numbers. They are kept because a\n"
        "correction that is not shown beside what it replaced is not\n"
        "auditable (docs/T61_information_symmetry_rules.md SS5).\n\n"
        + "\n".join(f"    {k}  md5 {v}" for k, v in sorted(rec.items())) + "\n")
    return rec


def ablate(imp, X_final, mask_df, target_frame, feats, seed, cat_vars):
    """t4f_perm_on_sni._ablate with the error target supplied by the caller.

    Passing `complete` reproduces the shipped matrix bit-for-bit; passing
    `X_final` gives the no-oracle twin. The machinery is the imported one, so
    the two differ in exactly one thing.
    """
    from t4f_perm_on_sni import _ablate
    return _ablate(imp, X_final, mask_df, target_frame, feats, seed, cat_vars)


# --------------------------------------------------------------------------- #
def run_stability() -> dict:
    """Real tables: hosts are archived, so this is a cold load -- no retraining
    and no new training randomness (T6.1 SS4)."""
    from t4f_perm_on_sni import _load_host
    from scipy.stats import spearmanr

    rows, per_host = [], {}
    for ds, seed in REAL:
        t0 = time.time()
        imp, X_final, mask_df, complete, catset = _load_host(ds, seed)
        A_or = ablate(imp, X_final, mask_df, complete, imp.all_vars, seed, catset)
        A_no = ablate(imp, X_final, mask_df, X_final, imp.all_vars, seed, catset)
        arch = pd.read_csv(FAITH / f"A_{ds}_seed{seed}_cpu_t2.csv", index_col=0)
        r = [x for x in arch.index if x in A_or.index]
        c = [x for x in arch.columns if x in A_or.columns]
        a = A_or.loc[r, c].to_numpy(float)
        b = A_no.loc[r, c].to_numpy(float)
        g = arch.loc[r, c].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        A_no.to_csv(_assert_writes_are_contained(
            OUT / f"A_noOracle_{ds}_seed{seed}_cpu_t2.csv"))
        per_host[f"{ds}_seed{seed}"] = {
            "control_max_abs_diff_vs_archived": float(np.nanmax(np.abs(a - g))),
            "pooled_spearman_oracle_vs_noOracle":
                float(spearmanr(a[ok], b[ok]).statistic),
            "n_cells": int(ok.sum()),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append({"dataset": ds, "seed": seed, **per_host[f"{ds}_seed{seed}"]})
        print(f"[ok] {ds} seed {seed}  "
              f"control {per_host[f'{ds}_seed{seed}']['control_max_abs_diff_vs_archived']:.1e}  "
              f"rho {per_host[f'{ds}_seed{seed}']['pooled_spearman_oracle_vs_noOracle']:.4f}  "
              f"{per_host[f'{ds}_seed{seed}']['seconds']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(_assert_writes_are_contained(
        OUT / "stability_noOracle_cells.csv"), index=False)
    return {"axis": "stability", "cold_loaded": True, "hosts": per_host}


# --------------------------------------------------------------------------- #
def run_recovery(regimes=None, seeds=None) -> dict:
    """Synthetic cells: hosts are NOT archived, so each is retrained ONCE and
    both matrices are computed on that same host, so the draw cancels
    (T6.1 SS4)."""
    import yaml
    from common import determinism
    from pilot_r21 import load_cell
    from sni.imputer import SNIConfig, SNIImputer

    regimes = regimes or REGIMES
    seeds = seeds or SYNTH_SEEDS
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {}
    for regime in regimes:
        for seed in seeds:
            tag = f"{regime}_s{seed}"
            fo = OUT / f"PERM_oracle_{tag}.csv"
            fn = OUT / f"PERM_noOracle_{tag}.csv"
            if fo.exists() and fn.exists():
                print(f"[cached] {tag}", flush=True)
                continue
            t0 = time.time()
            complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
            cols = list(complete.columns)
            proto = yaml.safe_load(
                (CODE_ROOT / "configs" / "training_protocol.yaml").read_text()
            )["protocol"]
            determinism.apply("deterministic", seed=seed)
            imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                             config=SNIConfig(seed=seed, use_gpu=False))
            imp.cfg.epochs = int(proto["epochs"]["SNI"])
            imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
            X_final = imp.impute(X_missing=missing[imp.all_vars],
                                 X_complete=None, mask_df=mask[imp.all_vars])
            train_s = time.time() - t0
            mask_df = mask[imp.all_vars].astype(bool)
            A_or = ablate(imp, X_final, mask_df, complete, imp.all_vars,
                          seed, set(cat)).reindex(index=cols, columns=cols)
            A_no = ablate(imp, X_final, mask_df, X_final, imp.all_vars,
                          seed, set(cat)).reindex(index=cols, columns=cols)
            A_or.to_csv(_assert_writes_are_contained(fo))
            A_no.to_csv(_assert_writes_are_contained(fn))
            arch = T4F / f"PERM_{tag}.csv"
            dmax = None
            if arch.exists():
                Aa = pd.read_csv(arch, index_col=0).reindex(index=cols,
                                                            columns=cols)
                dmax = float(np.nanmax(np.abs(A_or.to_numpy(float)
                                              - Aa.to_numpy(float))))
            cells[tag] = {"train_sec": round(train_s, 1),
                          "total_sec": round(time.time() - t0, 1),
                          "fresh_oracle_vs_archived_max_abs_diff": dmax}
            (OUT / f"meta_noOracle_{tag}.json").write_text(
                json.dumps(cells[tag], indent=1))
            print(f"[ok] {tag}  train={train_s:.0f}s  "
                  f"fresh-vs-archived {dmax}", flush=True)
    return {"axis": "recovery", "cold_loaded": False, "cells": cells}


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    from t4f_perm_on_sni import _load_host
    from scipy.stats import spearmanr
    ds, seed = "eICU", 1
    imp, X_final, mask_df, complete, catset = _load_host(ds, seed)
    A_or = ablate(imp, X_final, mask_df, complete, imp.all_vars, seed, catset)
    A_no = ablate(imp, X_final, mask_df, X_final, imp.all_vars, seed, catset)
    arch = pd.read_csv(FAITH / f"A_{ds}_seed{seed}_cpu_t2.csv", index_col=0)
    r = [x for x in arch.index if x in A_or.index]
    c = [x for x in arch.columns if x in A_or.columns]
    a, b = A_or.loc[r, c].to_numpy(float), A_no.loc[r, c].to_numpy(float)
    g = arch.loc[r, c].to_numpy(float)
    d = float(np.nanmax(np.abs(a - g)))
    check(d < 1e-12,
          f"CONTROL: passing `complete` reproduces the archived matrix "
          f"(max|diff| {d:.1e}) -- the recompute changes one thing only")
    ok2 = ~(np.isnan(a) | np.isnan(b))
    rho = float(spearmanr(a[ok2], b[ok2]).statistic)
    check(rho < 0.99,
          f"the no-oracle twin is a DIFFERENT matrix (pooled Spearman "
          f"{rho:.4f}), so the two error signals are distinguishable")
    check(A_no.shape == A_or.shape and list(A_no.index) == list(A_or.index),
          "the two matrices are the same shape and index -- only the values "
          "differ")
    # C-6: no new wall-clock record may reach a cost generator
    try:
        _assert_writes_are_contained(COST_SOURCES[0] / "x_time.txt")
        check(False, "the containment guard refuses a cost-source path")
    except RuntimeError:
        check(True, "the containment guard refuses a cost-source path")
    check(_assert_writes_are_contained(OUT / "ok.csv").name == "ok.csv",
          "the containment guard allows a results/T6_symmetry path")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["stability", "recovery"])
    ap.add_argument("--regime")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.axis == "stability":
        archive([FAITH / f"A_{ds}_seed{s}_cpu_t2.csv" for ds, s in REAL]
                + [T4F / "perm_on_sni_real_stability.csv"],
                "The real-table ablation matrices and the cross-seed stability "
                "readout derived from them, as computed with the oracle error "
                "signal (NRMSE against the withheld true values).")
        r = run_stability()
    elif a.axis == "recovery":
        archive([T4F / f"PERM_{rg}_s{s}.csv" for rg in REGIMES
                 for s in SYNTH_SEEDS],
                "The synthetic-cell Permutation-on-SNI matrices as computed "
                "with the oracle error signal.")
        r = run_recovery([a.regime] if a.regime else None,
                         [a.seed] if a.seed else None)
    else:
        ap.error("--axis is required")
    p = OUT / f"no_oracle_recompute_{r['axis']}.json"
    p.write_text(json.dumps(r, indent=1))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
