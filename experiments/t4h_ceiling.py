"""T4H.1 -- the reliability ceiling for faithfulness (self-attack analysis).

The strongest referee objection to "stability in excess of the host's own
reproducibility is a symptom of unfaithfulness": if the host's behavior is
only ~half reproducible, half of A is seed-specific, and a seed-invariant
artifact could still be faithfully capturing the REPRODUCIBLE half -- the low
faithfulness scores would then be the task's ceiling, not the artifact's
failure. This computes that ceiling and each object's attainment of it.

Classical disattenuation, all quantities at ONE caliber (matrix-level:
flattened target-row entries, the same selection the reliability quantities
use; the paper's Table 5 row-median caliber is reported alongside, never
mixed into the formulas):

    r_xx      within-seed reliability of A (fresh permutation draws)
    rho_AA    cross-seed consistency of A
    rho_XA    per-seed matrix-level Spearman(object, A), averaged over seeds
    rho_AA*   = rho_AA / r_xx          (true-score cross-seed consistency)
    ceiling   = sqrt(rho_AA*)          (best any seed-invariant artifact can do)
    rho*_XA   = rho_XA / sqrt(r_xx)    (disattenuated faithfulness)
    attain    = rho*_XA / ceiling  ==  rho_XA / sqrt(rho_AA)

Uncertainty: nonparametric bootstrap over SEEDS (the natural unit; n=5, so
intervals are honest and wide), 2000 draws, percentile 95% CIs, jointly for
ceiling and attainment. Approximation stated, not packaged as exact
inference: disattenuation assumes additive independent error in a Pearson
framework; we apply it to Spearman correlations.

Permutation-on-SNI is listed with attainment "identically 1 (circular)" --
its agreement with A is definitional, which is exactly why the faithfulness
axis excludes it.

    env PYTHONHASHSEED=2025 python experiments/t4h_ceiling.py
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
OUT = CODE_ROOT / "results" / "T4_ceiling"

DATASETS = ["MIMIC", "eICU"]
SEEDS = [1, 2, 3, 5, 8]
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest"]
N_BOOT = 2000


def _flat(M: pd.DataFrame, sel: np.ndarray) -> np.ndarray:
    return M.to_numpy(dtype=float)[sel]


def _sp(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def load_all(ds: str):
    A = {s: pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv", index_col=0)
         for s in SEEDS}
    G = {s: {g: pd.read_csv(T4F / f"A_G{g}_{ds}_seed{s}_cpu_t2.csv",
                            index_col=0) for g in (1, 2)} for s in SEEDS}
    feats = list(A[SEEDS[0]].columns)
    sel = A[SEEDS[0]].notna().to_numpy()
    X = {}
    for s in SEEDS:
        X[("SNI-D", s)] = pd.read_csv(
            FAITH / f"D_retrained_{ds}_seed{s}_cpu_t2.csv", index_col=0
        ).reindex(index=A[s].index, columns=feats)
        for m in OBJECTS[2:]:
            X[(m, s)] = pd.read_csv(
                FIVEWAY / f"D_{ds}_seed{s}_{m}.csv", index_col=0
            ).reindex(index=A[s].index, columns=feats)
    P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                    ).reindex(index=A[SEEDS[0]].index, columns=feats)
    for s in SEEDS:
        X[("P", s)] = P
    return A, G, X, sel


def _rowranks(M: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    """Per-row ranks over ref's non-NaN source set (constant: self column).

    Ranking each matrix once turns every pairwise row-Spearman into a plain
    Pearson of cached rank rows -- the bootstrap then costs milliseconds
    instead of millions of scipy calls.
    """
    from scipy.stats import rankdata
    rows = []
    for f in ref.index:
        cols = list(ref.loc[f].dropna().index)
        rows.append(rankdata(M.loc[f, cols].to_numpy(float)))
    return np.asarray(rows)


def _rowcorr(Ra: np.ndarray, Rb: np.ndarray) -> list:
    a = Ra - Ra.mean(axis=1, keepdims=True)
    b = Rb - Rb.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return list(num / den)


def rank_cache(A, G, X, seeds):
    ref = A[seeds[0]]
    # The cache's premise, asserted rather than assumed (P4-I follow-up): the
    # optimization replaced a reference implementation, and its correctness
    # rests on (i) every A/G matrix sharing ref's exact NaN pattern (the self
    # column per row) and (ii) every X matrix being NaN-free on ref's
    # selected columns. A silent violation would rank the wrong entries.
    refpat = ref.isna()
    for s in seeds:
        assert A[s].isna().equals(refpat), f"A NaN pattern differs (seed {s})"
        for g in (1, 2):
            assert G[s][g].isna().equals(refpat),                 f"G{g} NaN pattern differs (seed {s})"
    for o in OBJECTS:
        for s in seeds:
            M = X[(o, s)]
            for f in ref.index:
                cols = list(ref.loc[f].dropna().index)
                assert not M.loc[f, cols].isna().any(),                     f"X({o},{s}) has NaN inside ref's selection at row {f}"
    RA = {s: _rowranks(A[s], ref) for s in seeds}
    RG = {s: {g: _rowranks(G[s][g], ref) for g in (1, 2)} for s in seeds}
    RX = {(o, s): _rowranks(X[(o, s)], ref) for o in OBJECTS for s in seeds}
    return RA, RG, RX


def quantities_row(RA, RG, RX, seeds):
    """Row-level caliber -- identical unit and aggregation to Table 5:
    per-(target, unit) row correlations pooled, then the MEDIAN (P4-I)."""
    r_xx = []
    for s in seeds:
        for Ra, Rb in ((RA[s], RG[s][1]), (RA[s], RG[s][2]),
                       (RG[s][1], RG[s][2])):
            r_xx += _rowcorr(Ra, Rb)
    rho_AA = []
    for a, b in combinations(seeds, 2):
        rho_AA += _rowcorr(RA[a], RA[b])
    rho_XA = {}
    for o in OBJECTS:
        vals = []
        for s in seeds:
            vals += _rowcorr(RA[s], RX[(o, s)])
        rho_XA[o] = float(np.median(vals))
    return (float(np.median(r_xx)), float(np.median(rho_AA)), rho_XA)


def quantities(A, G, X, sel, seeds):
    """All three inputs on the seed subset, matrix-level, one caliber."""
    r_xx = [_sp(_flat(A[s], sel), _flat(G[s][g], sel))
            for s in seeds for g in (1, 2)]
    r_xx += [_sp(_flat(G[s][1], sel), _flat(G[s][2], sel)) for s in seeds]
    rho_AA = [_sp(_flat(A[a], sel), _flat(A[b], sel))
              for a, b in combinations(seeds, 2)]
    rho_XA = {o: [_sp(_flat(X[(o, s)], sel), _flat(A[s], sel))
                  for s in seeds] for o in OBJECTS}
    return (float(np.mean(r_xx)), float(np.mean(rho_AA)),
            {o: float(np.mean(v)) for o, v in rho_XA.items()})


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set (B48).",
              file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260821)
    out = {}
    rows = []
    faith = json.loads((FAITH / "faithfulness_summary.json").read_text())
    for ds in DATASETS:
        A, G, X, sel = load_all(ds)
        RA, RG, RX = rank_cache(A, G, X, SEEDS)
        blk = {}
        for caliber, qfn in (("row", None), ("matrix", None)):
            if caliber == "row":
                r_xx, rho_AA, rho_XA = quantities_row(RA, RG, RX, SEEDS)
            else:
                r_xx, rho_AA, rho_XA = quantities(A, G, X, sel, SEEDS)
            ceiling = float(np.sqrt(rho_AA / r_xx))
            boots = {"ceiling": [], **{o: [] for o in OBJECTS}}
            for _ in range(N_BOOT):
                bs = list(rng.choice(SEEDS, size=len(SEEDS), replace=True))
                if len(set(bs)) < 2:
                    continue
                uniq = list(dict.fromkeys(bs))
                if caliber == "row":
                    rb, ab, xb = quantities_row(RA, RG, RX, uniq)
                else:
                    rb, ab, xb = quantities(A, G, X, sel, uniq)
                if ab <= 0 or rb <= 0:
                    continue
                boots["ceiling"].append(np.sqrt(ab / rb))
                for o in OBJECTS:
                    boots[o].append(xb[o] / np.sqrt(ab))
            ci = {k: (float(np.percentile(v, 2.5)),
                      float(np.percentile(v, 97.5)))
                  for k, v in boots.items() if len(v)}
            cblk = {"r_xx": round(r_xx, 4), "rho_AA": round(rho_AA, 4),
                    "rho_AA_star": round(rho_AA / r_xx, 4),
                    "ceiling": round(ceiling, 4),
                    "ceiling_ci95": [round(c, 3) for c in ci["ceiling"]],
                    "n_boot_kept": len(boots["ceiling"])}
            if caliber == "row":
                # cross-check: row-level rho_XA must equal Table 5's medians
                for o, key in (("SNI-D", "SNI-D"), ("P", "P")):
                    want = faith[ds][key]["rho_median"]
                    got = rho_XA[o]
                    if abs(got - want) > 5e-4:
                        print(f"WARNING: row-level rho_XA({o}) {got:.4f} != "
                              f"Table-5 median {want:.4f} ({ds})",
                              file=sys.stderr)
            for o in OBJECTS:
                att = rho_XA[o] / np.sqrt(rho_AA)
                cblk[o] = {"rho_XA": round(rho_XA[o], 4),
                           "rho_XA_disattenuated": round(
                               rho_XA[o] / np.sqrt(r_xx), 4),
                           "attainment": round(float(att), 4),
                           "attainment_ci95": [round(c, 3) for c in ci[o]]}
                if caliber == "row":
                    rows.append({"dataset": ds, "object": o,
                                 "rho_XA": round(rho_XA[o], 4),
                                 "attainment": round(float(att), 4),
                                 "att_lo": round(ci[o][0], 3),
                                 "att_hi": round(ci[o][1], 3)})
            blk[caliber] = cblk
        rows.append({"dataset": ds, "object": "Permutation-on-SNI",
                     "rho_XA": 1.0, "attainment": float("nan"),
                     "att_lo": float("nan"), "att_hi": float("nan")})
        out[ds] = blk
    out["_method_note"] = (
        "Classical disattenuation applied to Spearman correlations; assumes "
        "additive independent error (Pearson framework). Approximate by "
        "construction. PRIMARY caliber is row-level (per-target row "
        "correlations pooled, then the median -- the exact unit and "
        "aggregation of Table 5, so its rho_XA are Table 5's numbers and a "
        "reader can re-derive attainment by hand); matrix-level kept as a "
        "robustness check (uniformly lower). Attainment "
        "algebraically reduces to rho_XA/sqrt(rho_AA); r_xx is shown because "
        "the ceiling itself depends on it. Bootstrap resamples seeds (n=5), "
        "2000 draws, percentile CIs; draws with <2 distinct seeds or "
        "non-positive inputs dropped and counted.")
    (OUT / "ceiling_analysis.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "ceiling_attainment.csv", index=False)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
