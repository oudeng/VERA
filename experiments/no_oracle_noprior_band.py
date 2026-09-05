"""The no-prior host band under information symmetry (T6.1 addendum 2026-08-29d SS3).

The sixth review's third finding: the with-prior host band was recomputed
without the privileged error signal, but the no-prior arm's band -- 0.489 on
eICU, 0.478 on MIMIC -- was not, and the manuscript's "no longer over-stable"
and "if anything conservative" rest on where the no-prior D sits relative to
it. Inferring this arm's direction from the other arm's is not evidence, so
this measures it.

The no-prior hosts are archived (models_NP_*.pt, Xfinal_NP_*), so this is a
COLD LOAD: no retraining, no new training randomness. Three things are
computed per host, with the same machinery the with-prior recompute used:

    A_oracle     the ablation with the error signal from the withheld values;
                 a CONTROL -- it must reproduce the archived A_NP_* matrix
    A_noOracle   the same ablation with the error signal from the host's own
                 completed table
    the band     stage_real's algorithm verbatim: pairwise Spearman over the
                 entries the FIRST seed's notna mask selects, all pairs

If the control does not reproduce the archived matrix, nothing is reported and
the failure is.

    env PYTHONHASHSEED=2025 python experiments/no_oracle_noprior_band.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
OUT = CODE_ROOT / "results" / "T6_symmetry"
DATASETS = ["MIMIC", "eICU"]
SEEDS15 = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
#: the five-seed caliber the ESM's no-prior panel reports beside the 15-seed one
SEEDS5 = [1, 2, 3, 5, 8]


def _contained(p: Path) -> Path:
    """T6.1: this recompute writes inside results/T6_symmetry/ and nowhere else."""
    rp = Path(p).resolve()
    if OUT.resolve() not in rp.parents:
        raise RuntimeError(f"refusing to write outside {OUT}: {rp}")
    return rp


def _band(A_by_seed: dict) -> dict:
    """t4f_perm_on_sni.stage_real's algorithm, verbatim caliber."""
    from scipy.stats import spearmanr
    seeds = sorted(A_by_seed)
    sel = A_by_seed[seeds[0]].notna().to_numpy()
    rows = []
    for a, b in combinations(seeds, 2):
        A = A_by_seed[a].to_numpy(float)[sel]
        B = A_by_seed[b].to_numpy(float)[sel]
        rows.append({"a": a, "b": b,
                     "spearman": round(float(spearmanr(A, B).statistic), 4)})
    v = [r["spearman"] for r in rows]
    return {"pairs": rows, "n_pairs": len(rows), "mean": round(float(np.mean(v)), 4),
            "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}


def run(datasets=None, seeds=None) -> dict:
    from no_oracle_recompute import ablate
    from t4f_perm_on_sni import _load_host

    datasets = datasets or DATASETS
    seeds = seeds or SEEDS15
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"rules": "docs/T61_information_symmetry_rules.md addendum 2026-08-29d SS3",
           "algorithm": "t4f_perm_on_sni.stage_real verbatim; hosts cold-loaded "
                        "from models_NP_*.pt, no retraining",
           "seeds": seeds, "datasets": {}}
    for ds in datasets:
        A_or, A_no, control = {}, {}, {}
        for s in seeds:
            t0 = time.time()
            imp, X_final, mask_df, complete, catset = _load_host(ds, s, prefix="NP_")
            a_or = ablate(imp, X_final, mask_df, complete, imp.all_vars, s, catset)
            a_no = ablate(imp, X_final, mask_df, X_final, imp.all_vars, s, catset)
            arch = pd.read_csv(FAITH / f"A_NP_{ds}_seed{s}_cpu_t2.csv", index_col=0)
            r = [x for x in arch.index if x in a_or.index]
            c = [x for x in arch.columns if x in a_or.columns]
            d = float(np.nanmax(np.abs(a_or.loc[r, c].to_numpy(float)
                                       - arch.loc[r, c].to_numpy(float))))
            control[s] = d
            A_or[s], A_no[s] = a_or.loc[r, c], a_no.loc[r, c]
            a_no.to_csv(_contained(OUT / f"A_NP_noOracle_{ds}_seed{s}_cpu_t2.csv"))
            print(f"[ok] NP {ds} seed{s}  control_max_abs_diff={d:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        worst = max(control.values())
        if worst > 1e-9:
            raise RuntimeError(
                f"{ds}: the oracle control did NOT reproduce the archived A_NP "
                f"matrices (max abs diff {worst:.3e}). Reporting nothing and "
                f"reporting this instead, per the addendum.")
        rec = {"control_max_abs_diff_vs_archived": worst,
               "band_oracle_15seed": _band(A_or),
               "band_noOracle_15seed": _band(A_no),
               "band_oracle_5seed": _band({s: A_or[s] for s in SEEDS5 if s in A_or}),
               "band_noOracle_5seed": _band({s: A_no[s] for s in SEEDS5 if s in A_no})}
        rec["delta_mean_15seed"] = round(rec["band_noOracle_15seed"]["mean"]
                                         - rec["band_oracle_15seed"]["mean"], 4)
        out["datasets"][ds] = rec
        print(f"[band] NP {ds}: oracle {rec['band_oracle_15seed']['mean']:.4f} "
              f"-> no-oracle {rec['band_noOracle_15seed']['mean']:.4f} "
              f"(15 seeds)", flush=True)
    p = _contained(OUT / "no_oracle_noprior_band.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    return out


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    for ds in DATASETS:
        miss = [s for s in SEEDS15
                if not (FAITH / f"models_NP_{ds}_seed{s}_cpu_t2.pt").exists()
                or not (FAITH / f"Xfinal_NP_{ds}_seed{s}_cpu_t2.csv").exists()
                or not (FAITH / f"A_NP_{ds}_seed{s}_cpu_t2.csv").exists()]
        chk(not miss, f"{ds}: all 15 no-prior hosts are archived ({miss[:3]})")

    from t4f_perm_on_sni import _load_host
    import inspect
    chk("prefix" in inspect.signature(_load_host).parameters,
        "the arm is a parameter of the ONE loader, not a second copy of it")

    from no_oracle_recompute import ablate
    chk(callable(ablate), "the ablation machinery is the imported one")

    # The band function must reproduce the SHIPPED no-prior band from the
    # archived matrices. Without this the recompute could differ from the
    # published number for a reason that has nothing to do with the oracle.
    import json as _j
    v = _j.loads((CODE_ROOT / "results" / "T4_noprior"
                  / "t43_verdict.json").read_text())
    shipped5 = float(v["filing2_noprior_own_host_band"]["mean"])
    arch = {s: pd.read_csv(FAITH / f"A_NP_eICU_seed{s}_cpu_t2.csv", index_col=0)
            for s in SEEDS5}
    got5 = _band(arch)["mean"]
    chk(abs(got5 - shipped5) <= 1e-3,
        f"the band function reproduces the shipped 5-seed no-prior band "
        f"({got5:.4f} vs {shipped5:.4f})")
    obs = _j.loads((CODE_ROOT / "results" / "T4_noprior"
                    / "t43_observables_15seed.json").read_text())
    shipped15 = float(obs["noprior_own_host_band_mean_15seed"])
    arch15 = {s: pd.read_csv(FAITH / f"A_NP_eICU_seed{s}_cpu_t2.csv", index_col=0)
              for s in SEEDS15}
    got15 = _band(arch15)["mean"]
    chk(abs(got15 - shipped15) <= 1e-3,
        f"and the shipped 15-seed one ({got15:.4f} vs {shipped15:.4f})")

    try:
        _contained(FAITH / "x.csv")
        chk(False, "the write guard refuses a path outside T6_symmetry")
    except RuntimeError:
        chk(True, "the write guard refuses a path outside T6_symmetry")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    run(a.datasets, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
