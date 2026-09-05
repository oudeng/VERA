"""T2g.1 -- how much of D's stability is inherited from the prior P?

The alternative explanation this rules in or out: P is a deterministic function
of the (roughly completed) table, so it is near-perfectly stable across seeds,
threads and devices *by construction*. If D is approximately P plus noise, then
"D is stable" (T2f.1: rho 0.89-0.95) means only "the correlation matrix is
stable" -- computable in milliseconds without training SNI. That is R2-2's
objection verbatim, and it sets the floor of the concession branch 1 can make.

**What P is here.** The prior an auditor could compute WITHOUT training any
neural network: SNI's own first-iteration prior. `impute()` computes it at
`imputer.py:317` from `X_current`, which at EM iteration 1 is the
IterativeImputer completion of the missing table (`_initial_stat_impute`,
no CPFA involved). This file reproduces that object by calling the *same
methods on the same class* -- `impute()` with `cfg.max_iters = 0` returns
exactly the cast initial completion and trains nothing; the prior then comes
from `_compute_correlation_prior` / `_extract_feature_prior` /
`_normalize_prior` verbatim, assembled row-wise into the same row-stochastic
d x d layout as D.

Two deliberate deviations from the instrument's wording, both recorded:

* The instrument parenthesises the prior recipe as "Pearson / Cramer's V /
  correlation ratio". The implementation SNI actually regularises with is
  one-hot encoding + |Pearson| + mean pooling (`imputer.py:589-663`); the
  binding requirement is "same recipe as sni internal", so the code wins.
* The instrument says "from each run's completed table". d_stability.py saved
  D and metrics but not completed tables, and recovering the *final* EM table
  would mean re-running 10 x 200-epoch trainings -- while the instrument
  budgets 2 h and "no model re-runs". The first-iteration prior is used
  instead, and it is the *right* object for the question: it is the only P in
  the pipeline that is model-free, i.e. the only one an auditor gets for free.

Partial correlation. For a seed pair (a, b): Spearman-partial by residualising
rank(vec(D_a)) on rank(vec(P_a)) and rank(vec(D_b)) on rank(vec(P_b)) (each
side on its own P; the P's are near-identical), then Pearson on the residuals.
Off-diagonal entries only, matching T2f's stability metric.

Synthetic three-way (instrument step 4): on the pilot's 15 cells score, against
the same ground truth and with the pilot's own `score()`/`measured_rows()`
imported unchanged: (i) P alone, (ii) SNI-D (re-scored from the saved matrices
as a harness sanity check against pilot_full_cells.csv), (iii) the residual of
D regressed on P -- does CPFA's reweighting of the prior carry recoverable
dependency information? Residuals are signed, and the sign matters (positive =
D upweights beyond the prior), so they are scored raw as well as |.|-ised.

    env PYTHONHASHSEED=2025 python experiments/prior_attribution.py --stage priors-real
    env PYTHONHASHSEED=2025 SNI_NUM_THREADS=1  python experiments/prior_attribution.py --stage priors-arm --arm cpu_t1
    env PYTHONHASHSEED=2025 SNI_NUM_THREADS=8  python experiments/prior_attribution.py --stage priors-arm --arm cpu_t8
    env PYTHONHASHSEED=2025 SNI_NUM_THREADS=24 python experiments/prior_attribution.py --stage priors-arm --arm cpu_t24
    env PYTHONHASHSEED=2025 python experiments/prior_attribution.py --stage priors-arm --arm cuda
    env PYTHONHASHSEED=2025 python experiments/prior_attribution.py --stage priors-synth
    env PYTHONHASHSEED=2025 python experiments/prior_attribution.py --stage analyze
"""

from __future__ import annotations

import os

# Thread count is a controlled variable (B84); set before numpy/torch. The
# priors of the perturbation arms are computed under that arm's thread count so
# the conditioning variable saw the same BLAS the run saw.
_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

STAB = CODE_ROOT / "results" / "T2f_d_stability"
PILOT = CODE_ROOT / "results" / "T2.5_pilot"
OUT = CODE_ROOT / "results" / "T2g_prior_attribution"

DATASETS = ["MIMIC", "eICU"]
SEEDS = [1, 2, 3, 5, 8]
ARMS = ["cpu_t1", "cpu_t8", "cpu_t24", "cuda"]      # T2f.2's four arms, all seed 1
REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SYNTH_SEEDS = [2025, 2026, 2027, 2028, 2029]


# --------------------------------------------------------------------------- #
# P -- via SNI's own methods, nothing re-implemented
# --------------------------------------------------------------------------- #
def compute_P(cat_vars, cont_vars, seed: int, missing: pd.DataFrame,
              mask_df: pd.DataFrame | None) -> pd.DataFrame:
    from sni.imputer import SNIConfig, SNIImputer

    imp = SNIImputer(categorical_vars=list(cat_vars),
                     continuous_vars=list(cont_vars),
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.max_iters = 0        # EM loop never entered: impute() returns the
    #                              cast initial completion and trains nothing.
    X0 = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    Prior_matrix, dims, cols = imp._compute_correlation_prior(X0)

    d = len(imp.all_vars)
    P = np.zeros((d, d), dtype=float)
    for i, f in enumerate(imp.all_vars):
        vec = imp._normalize_prior(
            imp._extract_feature_prior(Prior_matrix, f, cols, dims))
        others = [v for v in imp.all_vars if v != f]
        for val, o in zip(vec, others):
            P[i, imp.all_vars.index(o)] = float(val)
    return pd.DataFrame(P, index=imp.all_vars, columns=imp.all_vars)


def load_real_case(dataset: str):
    """Table, MAR@30 mask and schema exactly as d_stability.run_cell loads them."""
    from baselines.schema import DataSchema

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{dataset}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / dataset / f"{dataset}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    missing = complete[feats].mask(mask_df)
    return missing, mask_df, list(schema.categorical_vars), list(schema.continuous_vars)


def load_synth_case(regime: str, seed: int):
    from pilot_r21 import load_cell
    complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
    return complete, missing, mask, G, cat, cont


# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #
def offv(M: pd.DataFrame) -> np.ndarray:
    A = M.to_numpy(dtype=float)
    return A[~np.eye(len(A), dtype=bool)]


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def _resid_on(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
    return x - Z @ beta


def partial_corr(dA, dB, pA, pB, ranks: bool) -> float:
    """corr(D_a, D_b | P), each side residualized on its own (near-identical) P."""
    from scipy.stats import pearsonr, rankdata
    if ranks:
        dA, dB, pA, pB = (rankdata(v) for v in (dA, dB, pA, pB))
    return float(pearsonr(_resid_on(dA, pA), _resid_on(dB, pB)).statistic)


def cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def score_raw(S: pd.DataFrame, G: pd.DataFrame, keep: np.ndarray) -> dict:
    """The pilot's score() with the row_normalise(|.|) step removed, for signed
    scores (regression residuals). Same y, same cells, same metrics."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    A = S.to_numpy(dtype=float)
    T = G.to_numpy(dtype=float)
    off = ~np.eye(len(A), dtype=bool)
    sel = off & keep[:, None]
    y = (T[sel] > 0).astype(int)
    s = A[sel]
    if y.sum() == 0 or y.sum() == y.size:
        return {"auroc": np.nan, "auprc": np.nan, "prec_at_k": np.nan, "shd": np.nan}
    k = int(y.sum())
    order = np.argsort(-s)
    topk = np.zeros_like(y)
    topk[order[:k]] = 1
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "prec_at_k": float(y[order[:k]].mean()),
            "shd": int(np.sum(topk != y))}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_priors_real() -> None:
    for ds in DATASETS:
        missing, mask_df, cat, cont = load_real_case(ds)
        for s in SEEDS:
            tag = f"{ds}_seed{s}_cpu_t{_NT}"
            fp = OUT / f"P_{tag}.csv"
            if fp.exists():
                print(f"[cached] P_{tag}")
                continue
            P = compute_P(cat, cont, s, missing, mask_df)
            P.to_csv(fp)
            print(f"[ok] P_{tag}  rows={len(P)}", flush=True)


def stage_priors_arm(arm: str) -> None:
    want = {"cpu_t1": 1, "cpu_t8": 8, "cpu_t24": 24, "cuda": 2}[arm]
    if int(_NT) != want:
        print(f"REFUSING: arm {arm} ran under {want} BLAS threads, this process "
              f"has {_NT}. Launch with SNI_NUM_THREADS={want}.", file=sys.stderr)
        raise SystemExit(2)
    # P has no device dependence (sklearn/numpy only); the cuda arm's P differs
    # from cpu_t2's only in name. Computed anyway so every D has a P of record.
    missing, mask_df, cat, cont = load_real_case("MIMIC")
    tag = f"MIMIC_perturb_{arm}"
    P = compute_P(cat, cont, 1, missing, mask_df)
    P.to_csv(OUT / f"P_{tag}.csv")
    print(f"[ok] P_{tag} under {_NT} threads", flush=True)


def stage_priors_synth() -> None:
    for regime in REGIMES:
        for s in SYNTH_SEEDS:
            fp = OUT / f"P_synth_{regime}_s{s}.csv"
            if fp.exists():
                print(f"[cached] {fp.name}")
                continue
            complete, missing, mask, G, cat, cont = load_synth_case(regime, s)
            P = compute_P(cat, cont, s, missing[list(missing.columns)], None)
            P = P.reindex(index=list(complete.columns),
                          columns=list(complete.columns))
            P.to_csv(fp)
            print(f"[ok] {fp.name}", flush=True)


def _load_pair_matrices(ds: str):
    D, P = {}, {}
    for s in SEEDS:
        tag = f"{ds}_seed{s}_cpu_t2"
        D[s] = pd.read_csv(STAB / f"D_{tag}.csv", index_col=0)
        P[s] = pd.read_csv(OUT / f"P_{tag}.csv", index_col=0).reindex(
            index=D[s].index, columns=D[s].columns)
        assert not P[s].isna().any().any(), f"P/{tag}: index mismatch with D"
    return D, P


def stage_analyze() -> int:
    pair_rows, seed_rows = [], []

    # ---- real tables: the four quantities -------------------------------- #
    for ds in DATASETS:
        D, P = _load_pair_matrices(ds)
        for s in SEEDS:
            d, p = offv(D[s]), offv(P[s])
            rho_dp, _ = spearman(d, p)
            seed_rows.append({
                "dataset": ds, "seed": s,
                "rho_D_P_spearman": round(rho_dp, 4),
                "cos_D_P": round(cosine(d, p), 4),
                "relF_D_minus_P": round(float(np.linalg.norm(d - p)
                                              / np.linalg.norm(p)), 4)})
        for a, b in combinations(SEEDS, 2):
            dA, dB, pA, pB = offv(D[a]), offv(D[b]), offv(P[a]), offv(P[b])
            r_pp, _ = spearman(pA, pB)
            r_dd, _ = spearman(dA, dB)
            pair_rows.append({
                "group": f"{ds}_seed", "a": a, "b": b,
                "rho_P_P": round(r_pp, 4),
                "P_identical": bool(np.allclose(pA, pB, atol=1e-12)),
                "rho_D_D": round(r_dd, 4),
                "partial_rho_DD_given_P_spearman":
                    round(partial_corr(dA, dB, pA, pB, ranks=True), 4),
                "partial_rho_DD_given_P_pearson":
                    round(partial_corr(dA, dB, pA, pB, ranks=False), 4)})

    # ---- perturbation arms ----------------------------------------------- #
    Dp = {a: pd.read_csv(STAB / f"D_MIMIC_perturb_{a}.csv", index_col=0)
          for a in ARMS}
    Pp = {a: pd.read_csv(OUT / f"P_MIMIC_perturb_{a}.csv", index_col=0).reindex(
              index=Dp[a].index, columns=Dp[a].columns) for a in ARMS}
    for a, b in combinations(ARMS, 2):
        dA, dB, pA, pB = offv(Dp[a]), offv(Dp[b]), offv(Pp[a]), offv(Pp[b])
        grp = "MIMIC_perturb_thread" if "cuda" not in (a, b) else "MIMIC_perturb_device"
        r_pp, _ = spearman(pA, pB)
        r_dd, _ = spearman(dA, dB)
        pair_rows.append({
            "group": grp, "a": a, "b": b,
            "rho_P_P": round(r_pp, 4),
            "P_identical": bool(np.allclose(pA, pB, atol=1e-12)),
            "rho_D_D": round(r_dd, 4),
            "partial_rho_DD_given_P_spearman":
                round(partial_corr(dA, dB, pA, pB, ranks=True), 4),
            "partial_rho_DD_given_P_pearson":
                round(partial_corr(dA, dB, pA, pB, ranks=False), 4)})

    pairs = pd.DataFrame(pair_rows)
    perseed = pd.DataFrame(seed_rows)
    pairs.to_csv(OUT / "attribution_pairs.csv", index=False)
    perseed.to_csv(OUT / "attribution_perseed.csv", index=False)

    # cross-check rho_D_D against T2f's published pairs files
    for ds in DATASETS:
        ref = pd.read_csv(STAB / f"pairs_seed_{ds}.csv")
        got = pairs[pairs.group == f"{ds}_seed"].rho_D_D.to_numpy()
        want = ref.spearman.to_numpy()
        if not np.allclose(np.sort(got), np.sort(want), atol=5e-4):
            print(f"WARNING: rho_D_D mismatch vs pairs_seed_{ds}.csv", file=sys.stderr)

    # ---- synthetic three-way --------------------------------------------- #
    from pilot_r21 import measured_rows, score
    ref = pd.read_csv(PILOT / "pilot_full_cells.csv")
    synth_rows = []
    for regime in REGIMES:
        for s in SYNTH_SEEDS:
            complete, missing, mask, G, cat, cont = load_synth_case(regime, s)
            cols = list(complete.columns)
            mats = {m: pd.read_csv(
                        PILOT / f"D_{regime}_s{s}_{m}.csv", index_col=0
                    ).reindex(index=cols, columns=cols).fillna(0.0)
                    for m in ["SNI-D", "MissForest-importance",
                              "SHAP-on-MissForest", "Permutation-on-MissForest"]}
            common = np.ones(len(cols), dtype=bool)
            for M in mats.values():
                common &= measured_rows(M)

            Pm = pd.read_csv(OUT / f"P_synth_{regime}_s{s}.csv", index_col=0
                             ).reindex(index=cols, columns=cols)

            sc_D = score(mats["SNI-D"], G, keep=common)
            row_ref = ref[(ref.regime == regime) & (ref.seed == s)
                          & (ref.method == "SNI-D")].iloc[0]
            if abs(sc_D["auroc"] - float(row_ref.auroc)) > 1e-6:
                print(f"WARNING: SNI-D re-score {sc_D['auroc']:.6f} != pilot "
                      f"{row_ref.auroc:.6f} ({regime}/s{s})", file=sys.stderr)

            sc_P = score(Pm, G, keep=common)

            # residual of D on P over the scored cells, signed and |.|
            off = ~np.eye(len(cols), dtype=bool)
            sel = off & common[:, None]
            dv = mats["SNI-D"].to_numpy(float)[sel]
            pv = Pm.to_numpy(float)[sel]
            resid = _resid_on(dv, pv)
            R = np.zeros((len(cols), len(cols)))
            R[sel] = resid
            Rdf = pd.DataFrame(R, index=cols, columns=cols)
            sc_R = score_raw(Rdf, G, keep=common)
            sc_Rabs = score_raw(Rdf.abs(), G, keep=common)

            for name, sc in [("P-alone", sc_P), ("SNI-D", sc_D),
                             ("resid(D|P)_signed", sc_R),
                             ("resid(D|P)_abs", sc_Rabs)]:
                synth_rows.append({"regime": regime, "seed": s, "method": name,
                                   **{k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in sc.items()}})
            r_dp, _ = spearman(dv, pv)
            synth_rows.append({"regime": regime, "seed": s,
                               "method": "corr(D,P)_spearman",
                               "auroc": round(r_dp, 4)})

    synth = pd.DataFrame(synth_rows)
    synth.to_csv(OUT / "synth_threeway_cells.csv", index=False)

    # ---- summary + stop-condition flags ----------------------------------- #
    def _g(grp):
        return pairs[pairs.group == grp]

    summary = {}
    for ds in DATASETS:
        g = _g(f"{ds}_seed")
        s5 = perseed[perseed.dataset == ds]
        summary[ds] = {
            "rho_P_P_mean": round(float(g.rho_P_P.mean()), 4),
            "P_identical_all_pairs": bool(g.P_identical.all()),
            "rho_D_P_mean": round(float(s5.rho_D_P_spearman.mean()), 4),
            "cos_D_P_mean": round(float(s5.cos_D_P.mean()), 4),
            "rho_D_D_mean": round(float(g.rho_D_D.mean()), 4),
            "partial_rho_mean": round(float(
                g.partial_rho_DD_given_P_spearman.mean()), 4),
            "partial_rho_min": round(float(
                g.partial_rho_DD_given_P_spearman.min()), 4),
            "relF_D_minus_P_mean": round(float(s5.relF_D_minus_P.mean()), 4)}
    for grp in ["MIMIC_perturb_thread", "MIMIC_perturb_device"]:
        g = _g(grp)
        summary[grp] = {
            "rho_D_D_mean": round(float(g.rho_D_D.mean()), 4),
            "partial_rho_mean": round(float(
                g.partial_rho_DD_given_P_spearman.mean()), 4)}

    pv = synth[synth.method == "P-alone"].auroc.astype(float)
    dv = synth[synth.method == "SNI-D"].auroc.astype(float)
    summary["synth"] = {
        "P_alone_auroc_mean": round(float(pv.mean()), 4),
        "SNI_D_auroc_mean": round(float(dv.mean()), 4),
        "P_beats_or_ties_D": bool(pv.mean() >= dv.mean())}

    stop1 = any(summary[ds]["partial_rho_mean"] < 0.5 for ds in DATASETS)
    stop2 = summary["synth"]["P_beats_or_ties_D"]
    summary["stop_conditions"] = {
        "partial_rho_below_0.5": stop1,
        "synth_P_alone_recovery_geq_D": stop2,
        "triggered": bool(stop1 or stop2)}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["priors-real", "priors-arm", "priors-synth", "analyze"])
    ap.add_argument("--arm", choices=ARMS)
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    if a.stage == "priors-real":
        stage_priors_real()
    elif a.stage == "priors-arm":
        if not a.arm:
            ap.error("--arm required for priors-arm")
        stage_priors_arm(a.arm)
    elif a.stage == "priors-synth":
        stage_priors_synth()
    else:
        return stage_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
