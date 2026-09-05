"""T5.3 TAP baseline family (P5R-H SS7.1; rules: docs/T53_tap_family_rules.md,
commit 89c386d -- committed before any family artifact existed).

Five training-free variants computed from frozen inputs, read against the
archived ablation matrices under the probe-1 full-scope convention, and
summarized under the manuscript's unified T estimand. Prospectively
specified sensitivity layer; no verdict gated.

    env PYTHONHASHSEED=2025 python experiments/t53_tap_family.py
    env PYTHONHASHSEED=2025 python experiments/t53_tap_family.py --selftest
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

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
OUT = CODE_ROOT / "results" / "T5_family"
DATASETS = ["MIMIC", "eICU"]
ALL_SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
N_RANDOM = 20
MI_BINS = 8
OBS_MIN_PAIRS = 30
#: the seed of the archived TAP_0 matrix the family compares against;
#: the corrected variant inputs are rebuilt under the same seed
ARCHIVED_TAP_SEED = 1


def _schema(ds: str):
    from baselines.schema import DataSchema
    sc = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    return list(sc.categorical_vars), list(sc.continuous_vars)


def initial_completion_input(ds: str, seed: int = ARCHIVED_TAP_SEED
                             ) -> pd.DataFrame:
    """The table the archived TAP_0 was computed on, rebuilt the same way.

    experiments/prior_attribution.compute_P sets max_iters = 0, so impute()
    returns the cast MICE initial completion of the masked table and trains
    nothing. That is the fair input: it contains no withheld value. Verified
    per run by experiments/tap_lineage_audit.py, which classifies every
    masked coordinate of this table and requires CLEAN-COMPLETION.
    """
    from baselines.schema import DataSchema
    from sni.imputer import SNIConfig, SNIImputer
    G = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                    / f"{ds}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
    sc = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    feats = list(sc.categorical_vars) + list(sc.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(G.columns))[feats]
    missing = G[feats].mask(mask_df)
    imp = SNIImputer(categorical_vars=list(sc.categorical_vars),
                     continuous_vars=list(sc.continuous_vars),
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.max_iters = 0
    return imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)


def _pool_matrix(X: pd.DataFrame, cat: list, cont: list,
                 corr: str) -> pd.DataFrame:
    """The TAP recipe with a pluggable correlation: one-hot encode, NaN->0,
    |corr| over encoded columns, type-aware block pooling per target row
    (no row normalization) -- mirroring the archived P readout path."""
    from sklearn.preprocessing import OneHotEncoder
    from sni.imputer import SNIConfig, SNIImputer
    feats = cat + cont
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=0, use_gpu=False))
    Prior, dims, cols = imp._compute_correlation_prior(X[feats])
    if corr == "pearson":
        M = Prior                     # |Pearson| straight from the recipe
    elif corr == "spearman":
        # rebuild the SAME encoded matrix, then |Spearman|
        blocks, col_names = [], []
        for c in feats:
            if c in cat:
                ohe = OneHotEncoder(sparse_output=False,
                                    handle_unknown="ignore")
                arr = X[c].astype(str).values.reshape(-1, 1)
                ohe.fit(arr)
                col_names += [f"{c}__{v}" for v in ohe.categories_[0]]
                blocks.append(ohe.transform(arr))
            else:
                col_names.append(c)
                blocks.append(X[[c]].apply(pd.to_numeric,
                                           errors="coerce").values)
        X_enc = np.nan_to_num(np.hstack(blocks), nan=0.0)
        from scipy.stats import spearmanr
        rho = spearmanr(X_enc).statistic
        rho = np.atleast_2d(np.nan_to_num(rho, nan=0.0))
        M = np.abs(rho)
        cols = col_names
    else:
        raise ValueError(corr)
    C = pd.DataFrame(0.0, index=feats, columns=feats)
    for f in feats:
        vec = imp._extract_feature_prior(M, f, cols, dims)
        C.loc[f, [v for v in feats if v != f]] = vec
    return C


def _mi_matrix(X: pd.DataFrame, cat: list, cont: list) -> pd.DataFrame:
    """Normalized mutual information per feature pair; continuous columns
    discretized into MI_BINS quantile bins (rules-fixed), categoricals
    native; symmetric, diagonal zero."""
    from sklearn.metrics import normalized_mutual_info_score
    feats = cat + cont
    disc = {}
    for c in feats:
        if c in cont:
            disc[c] = pd.qcut(pd.to_numeric(X[c], errors="coerce"),
                              MI_BINS, labels=False, duplicates="drop")
        else:
            disc[c] = X[c].astype(str)
    C = pd.DataFrame(0.0, index=feats, columns=feats)
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            v = normalized_mutual_info_score(disc[a], disc[b])
            C.loc[a, b] = C.loc[b, a] = float(v)
    return C


def _observed_only_matrix(ds: str, cat: list, cont: list) -> tuple:
    """Pairwise-complete |Pearson| on the frozen MAR@30 masked table, NaN
    respected in the encoding (missing -> NaN across the one-hot block);
    entries with < OBS_MIN_PAIRS pairwise rows -> 0 (counted)."""
    from sni.imputer import SNIConfig, SNIImputer
    X = pd.read_csv(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                    / ds / f"{ds}_MAR_30per.csv")
    feats = cat + cont
    blocks, col_names, dims = [], [], {}
    for c in feats:
        col = X[c]
        if c in cat:
            vals = sorted(v for v in col.dropna().astype(str).unique())
            names = [f"{c}__{v}" for v in vals]
            dims[c] = names
            arr = np.full((len(X), len(vals)), np.nan)
            notna = col.notna().to_numpy()
            sv = col.astype(str).to_numpy()
            for j, v in enumerate(vals):
                arr[notna, j] = (sv[notna] == v).astype(float)
            blocks.append(arr)
            col_names += names
        else:
            dims[c] = [c]
            blocks.append(pd.to_numeric(col, errors="coerce"
                                        ).to_numpy().reshape(-1, 1))
            col_names.append(c)
    E = pd.DataFrame(np.hstack(blocks), columns=col_names)
    corr = E.corr(min_periods=OBS_MIN_PAIRS)
    n_thin = int(corr.isna().to_numpy().sum())
    corr = corr.fillna(0.0).abs()
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=0, use_gpu=False))
    C = pd.DataFrame(0.0, index=feats, columns=feats)
    M = corr.to_numpy()
    for f in feats:
        vec = imp._extract_feature_prior(M, f, col_names, dims)
        C.loc[f, [v for v in feats if v != f]] = vec
    return C, n_thin


def _rho(Mrow: np.ndarray, Arow: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(Mrow, Arow).statistic)


def _family_T(blocks: dict) -> dict:
    from experiments.t51_cluster_stats import seed_boot_ci_T, sign_flip_exact
    m1 = sign_flip_exact(blocks)
    lo, hi = seed_boot_ci_T(blocks)
    return {"n_seeds": m1["n_blocks"],
            "T": round(m1["observed_stat_mean_of_block_medians"], 6),
            "ci95_T": [round(lo, 6), round(hi, 6)],
            "p_exact": m1["p_two_sided"],
            "floor": m1["floor_two_sided"]}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, summary = [], {"rules": "docs/T53_tap_family_rules.md@89c386d",
                          "input_correction": "docs/T53_input_correction_rules.md (2026-08-29): the abs_spearman and mutual_information variants are computed on the initial completion, not the pre-mask table",
                          "params": {"n_random": N_RANDOM, "mi_bins": MI_BINS,
                                     "obs_min_pairs": OBS_MIN_PAIRS},
                          "datasets": {}}
    for d_idx, ds in enumerate(DATASETS):
        cat, cont = _schema(ds)
        feats = cat + cont
        # T5.3 input correction, 2026-08-29 (rules:
        # docs/T53_input_correction_rules.md). These two variants used to be
        # computed on data/derived_shuffled/{ds}_complete.csv -- the PRE-MASK
        # ground-truth table -- while the comparison baseline, the archived
        # TAP_0 matrix, was computed on the initial completion of the masked
        # table. The comparison was asymmetric and favored these variants.
        # They now read the same initial completion the archived TAP_0 used,
        # obtained by the same code path and the same seed.
        fair = initial_completion_input(ds)
        P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                        ).reindex(index=feats, columns=feats)
        variants = {"abs_spearman": _pool_matrix(fair, cat, cont,
                                                 "spearman"),
                    "mutual_information": _mi_matrix(fair, cat, cont)}
        obs, n_thin = _observed_only_matrix(ds, cat, cont)
        variants["observed_only_tap"] = obs
        rng_mats = {}
        for r in range(N_RANDOM):
            rng = np.random.default_rng(90_000 + 1_000 * d_idx + r)
            Mr = pd.DataFrame(rng.uniform(size=(len(feats), len(feats))),
                              index=feats, columns=feats)
            np.fill_diagonal(Mr.values, 0.0)
            rng_mats[r] = Mr

        seeds_found = [s for s in ALL_SEEDS
                       if (FAITH / f"A_{ds}_seed{s}_cpu_t2.csv").exists()]
        d_blocks = {v: {} for v in variants}
        r_blocks = {r: {} for r in rng_mats}
        for s in seeds_found:
            A = pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv", index_col=0)
            per_t = {v: [] and [] or [] for v in variants}
            per_t = {v: [] for v in variants}
            per_r = {r: [] for r in rng_mats}
            for f in A.index:
                srcs = [c for c in feats if c != f]
                Arow = A.loc[f, srcs].to_numpy(float)
                Prow = P.loc[f, srcs].to_numpy(float)
                rho_p = _rho(Prow, Arow)
                for v, M in variants.items():
                    rv = _rho(M.loc[f, srcs].to_numpy(float), Arow)
                    per_t[v].append(rv - rho_p)
                    cells.append({"dataset": ds, "seed": s, "target": f,
                                  "variant": v, "rho": round(rv, 6),
                                  "delta_vs_tap": round(rv - rho_p, 6)})
                for r, M in rng_mats.items():
                    per_r[r].append(_rho(M.loc[f, srcs].to_numpy(float),
                                         Arow) - rho_p)
            for v in variants:
                d_blocks[v][s] = per_t[v]
            for r in rng_mats:
                r_blocks[r][s] = per_r[r]

        ds_sum = {"n_seeds_found": len(seeds_found),
                  "uniform": {"degenerate": True,
                              "reason": "constant row induces no ranking; "
                                        "Spearman undefined (rules SS 'uniform')"}}
        for v in variants:
            ds_sum[v] = _family_T(d_blocks[v])
        rTs = sorted(_family_T(r_blocks[r])["T"] for r in rng_mats)
        ds_sum["random"] = {"n_replicates": N_RANDOM,
                            "T_min": rTs[0], "T_median": rTs[len(rTs) // 2],
                            "T_max": rTs[-1]}
        if "observed_only_tap" in ds_sum:
            ds_sum["observed_only_tap"]["n_thin_pairs_zeroed"] = n_thin
        summary["datasets"][ds] = ds_sum

    pd.DataFrame(cells).to_csv(OUT / "tapfam_cells.csv", index=False)
    (OUT / "tapfam_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200),
                      "b": rng.normal(size=200),
                      "c": rng.integers(0, 3, 200).astype(str)})
    X["b"] = X["a"] * 0.9 + rng.normal(scale=0.1, size=200)
    C = _pool_matrix(X, ["c"], ["a", "b"], "spearman")
    check(C.loc["a", "b"] > 0.8, "abs-spearman: strong monotone pair high")
    check(abs(C.loc["a", "c"]) < 0.4, "abs-spearman: independent cat low")
    M = _mi_matrix(X, ["c"], ["a", "b"])
    check(M.loc["a", "b"] > M.loc["a", "c"],
          "MI: dependent pair above independent pair")
    check(np.allclose(M.values, M.values.T), "MI symmetric")
    r1 = np.random.default_rng(90_000).uniform(size=3)
    r2 = np.random.default_rng(90_000).uniform(size=3)
    check(np.allclose(r1, r2), "random variant rng reproducible")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    s = run()
    for ds, d in s["datasets"].items():
        print(ds, json.dumps({k: v for k, v in d.items()
                              if k != "uniform"})[:300])
    print(f"[ok] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
