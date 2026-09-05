"""T3.2 -- is D faithful to SNI's own behavior? (decision rule: docs/
T32_faithfulness_decision_rule.md, commit 6030500 -- written and committed
before this harness ran.)

Ground truth is ablation-style: A[f,j] = how much SNI's imputation of target f
degrades when source j is destroyed at inference time. "Destroyed" = the
standardized model-input column j is permuted across rows (marginal preserved,
joint structure broken), 5 permutations averaged. The models are the run's own
trained per-feature CPFAs in eval mode; nothing about training, masks or the
pipeline is touched. In the default variant the model input has no mask
channel (asserted), so a permutation cannot signal "j was intervened on".

Because T2f.1 saved no weights, each run is retrained with byte-for-byte the
d_stability.run_cell recipe (same table, same fixed MAR@30 mask, same
protocol, CPU, threads pinned) -- and the retraining is verified, not assumed:
the recomputed dependency matrix must be bit-identical to the stored
`results/T2f_d_stability/D_{tag}.csv` (the environment was verified zero-drift
in P2g-PRE, and T2f.3 established bit-identical reruns). A non-identical D is
loudly flagged and recorded; the run still proceeds, because A is measured
against the retrained models themselves.

Inference mirrors `_train_continuous_feature`'s own prediction path
(`imputer.py:957-969`): encode with the imputer's `_encode_dataframe_for_training`,
standardize on target-observed rows, model.eval() + no_grad, inverse-transform
by the y scaler. The base table is the run's returned completion X_final
rather than the unsaved final-iteration X_current; baseline and ablated
predictions share every piece of that path, so A -- a difference -- is not
biased by it. Sanity check per run: the baseline NRMSE pooled over targets
must be within 50% of the run's recorded cont_NRMSE.

Permutation RNG (prospectively specified): np.random.default_rng(
    10_000*seed + 100*f_index + 5*j_index + r),  r in 0..4,
f_index/j_index = positions in the table's feature order.

    env PYTHONHASHSEED=2025 python experiments/faithfulness.py --stage train --datasets MIMIC
    env PYTHONHASHSEED=2025 python experiments/faithfulness.py --stage train --datasets eICU
    env PYTHONHASHSEED=2025 python experiments/faithfulness.py --stage analyze
"""

from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

STAB = CODE_ROOT / "results" / "T2f_d_stability"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
OUT = CODE_ROOT / "results" / "T3_faithfulness"

DATASETS = ["MIMIC", "eICU"]
SEEDS = [1, 2, 3, 5, 8]
N_PERM = 5
MF_METHODS = ["MissForest-importance", "SHAP-on-MissForest",
              "Permutation-on-MissForest"]


def run_one(ds: str, seed: int, variant: str = "SNI") -> None:
    import torch
    import yaml
    from sklearn.preprocessing import StandardScaler
    from baselines.schema import DataSchema
    from common import determinism
    from sni.imputer import SNIConfig, SNIImputer

    tag = (f"{ds}_seed{seed}_cpu_t{_NT}" if variant == "SNI"
           else f"NP_{ds}_seed{seed}_cpu_t{_NT}")
    if (OUT / f"A_{tag}.csv").exists():
        print(f"[cached] {tag}")
        return

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    missing = complete[feats].mask(mask_df)

    proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                            ).read_text())["protocol"]
    epochs = int(proto["epochs"]["SNI"])
    determinism.apply("deterministic", seed=seed)
    imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                     continuous_vars=list(schema.continuous_vars),
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.epochs = epochs
    imp.cfg.early_stopping_patience = epochs + 1
    imp.cfg.variant = variant          # T4.3: "NoPrior" -> alpha_f = 0, no prior loss
    assert imp.cfg.variant != "SNI-M", \
        "mask-aware variant would need the mask channel held fixed explicitly"

    t0 = time.time()
    X_final = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    wall = time.time() - t0

    D_retr = imp.compute_dependency_matrix()
    D_retr.to_csv(OUT / f"D_retrained_{tag}.csv")
    if variant == "SNI" and (STAB / f"D_{tag}.csv").exists():
        D_ref = pd.read_csv(STAB / f"D_{tag}.csv", index_col=0)
        dmax = float(np.abs(D_retr.to_numpy() - D_ref.to_numpy()).max())
        bit_identical = bool(dmax == 0.0)
        if not bit_identical:
            print(f"WARNING: retrained D differs from stored T2f matrix "
                  f"(max |diff| = {dmax:.3e}) for {tag} -- recorded, continuing "
                  f"against the retrained models", file=sys.stderr)
    elif variant == "SNI":
        # Expansion seeds (T51 addendum, seeds 13..987) postdate T2f: there
        # is no stored matrix to reproduce; D is fresh evidence, as for the
        # NoPrior arm. The training recipe above is byte-identical either way.
        print(f"[note] no stored T2f reference for {tag} (expansion seed)")
        dmax, bit_identical = float("nan"), None
    else:
        # No stored reference exists for the NoPrior variant; its D is fresh
        # evidence (stability axis), not a reproduction.
        dmax, bit_identical = float("nan"), None

    # Insurance against re-paying the training: weights + completion.
    torch.save({f: m.state_dict() for f, m in imp.models.items()},
               OUT / f"models_{tag}.pt")
    X_final.to_csv(OUT / f"Xfinal_{tag}.csv", index=False)

    targets = [f for f in feats if int(mask_df[f].sum()) > 0]
    A = pd.DataFrame(np.nan, index=targets, columns=feats)
    err0_rec, err0_range_rec = {}, {}
    n = len(X_final)

    for f in targets:
        fi = feats.index(f)
        Z_df = X_final.drop(columns=[f])
        srcs = list(Z_df.columns)
        Z_enc, _ = imp._encode_dataframe_for_training(Z_df)
        Z_enc = np.nan_to_num(Z_enc, nan=0.0)
        present = ~mask_df[f].to_numpy(dtype=bool)
        miss = np.where(~present)[0]
        scaler_Z = StandardScaler().fit(Z_enc[present])
        Z_s = scaler_Z.transform(Z_enc)
        y_obs = pd.to_numeric(X_final[f], errors="coerce").to_numpy(float)[present]
        scaler_y = StandardScaler().fit(y_obs.reshape(-1, 1))
        truth = pd.to_numeric(complete[f], errors="coerce").to_numpy(float)[miss]
        sd = float(truth.std()) or 1.0
        model = imp.models[f]
        model.eval()

        def _predict(Zmat: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                yh = model(torch.tensor(Zmat[miss], dtype=torch.float32))[0]
            return scaler_y.inverse_transform(
                yh.cpu().numpy().reshape(-1, 1)).flatten()

        rmse0 = float(np.sqrt(np.mean((truth - _predict(Z_s)) ** 2)))
        err0 = rmse0 / sd
        err0_rec[f] = err0
        # Range-normalized twin, for the sanity flag only: the pipeline's
        # recorded cont_NRMSE divides by the complete column's range
        # (evaluation/metrics.py), not by std. A[f,j] stays std-normalized as
        # prospectively specified; comparing unlike conventions is what made the first
        # sanity flags fire spuriously (diagnosed 08-21 05:44).
        col = pd.to_numeric(complete[f], errors="coerce")
        rng_f = float(col.max() - col.min()) or 1.0
        err0_range_rec[f] = rmse0 / rng_f

        for jj, j in enumerate(srcs):
            deltas = []
            for r in range(N_PERM):
                rng = np.random.default_rng(10_000 * seed + 100 * fi + 5 * jj + r)
                Zp = Z_s.copy()
                Zp[:, jj] = Zp[rng.permutation(n), jj]
                e = float(np.sqrt(np.mean((truth - _predict(Zp)) ** 2)) / sd)
                deltas.append(e - err0)
            A.loc[f, j] = float(np.mean(deltas))

    A.to_csv(OUT / f"A_{tag}.csv")

    base_pooled = float(np.mean(list(err0_rec.values())))
    base_range = float(np.mean(list(err0_range_rec.values())))
    if variant == "SNI" and (STAB / f"M_{tag}.json").exists():
        m_ref = json.loads((STAB / f"M_{tag}.json").read_text())
        rec_nrmse = float(m_ref.get("cont_NRMSE", float("nan")))
        sane = bool(abs(base_range - rec_nrmse) <= 0.5 * rec_nrmse)
        if not sane:
            print(f"WARNING: baseline NRMSE(range) {base_range:.4f} far from "
                  f"recorded {rec_nrmse:.4f} for {tag}", file=sys.stderr)
    else:
        rec_nrmse, sane = float("nan"), None   # no stored reference
        #    (NoPrior arm, or a T51-addendum expansion seed postdating T2f)
    (OUT / f"meta_{tag}.json").write_text(json.dumps(
        {"tag": tag, "variant": variant, "wall_train_sec": round(wall, 1),
         "D_bit_identical_to_T2f": bit_identical, "D_max_abs_diff": dmax,
         "baseline_nrmse_pooled_std": round(base_pooled, 6),
         "baseline_nrmse_pooled_range": round(base_range, 6),
         "recorded_cont_NRMSE": rec_nrmse, "baseline_sane": sane,
         "n_targets": len(targets), "n_perm": N_PERM}, indent=2))
    print(f"[ok] {tag}  train={wall:.0f}s  D_bitwise={bit_identical}  "
          f"base_NRMSE range={base_range:.4f} (recorded {rec_nrmse}) "
          f"std={base_pooled:.4f}", flush=True)


# --------------------------------------------------------------------------- #
def _redundant_set(ds: str):
    """T3.2-R1 (docs/T32_R1_redundancy_precheck.md, prospectively specified).

    Feature-level pooled |Pearson| on the COMPLETE table, computed with the
    prior's own type-aware recipe (`_compute_correlation_prior` +
    `_extract_feature_prior`, no row normalization). R = features having any
    off-self pooled correlation > 0.8 (threshold fixed in the prospective specification).
    """
    from baselines.schema import DataSchema
    from sni.imputer import SNIConfig, SNIImputer

    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    cat = list(schema.categorical_vars)
    cont = list(schema.continuous_vars)
    feats = cat + cont
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=0, use_gpu=False))
    Prior_matrix, dims, cols = imp._compute_correlation_prior(complete[feats])
    C = pd.DataFrame(0.0, index=feats, columns=feats)
    for f in feats:
        vec = imp._extract_feature_prior(Prior_matrix, f, cols, dims)
        others = [v for v in feats if v != f]
        C.loc[f, others] = vec
    R = {f for f in feats if float(C.loc[f].drop(f).max()) > 0.8}
    return R, C


def _paired_effect(delta: np.ndarray) -> dict:
    """Effect size + CI for a paired delta vector (first-author directive,
    2026-08-20, reporting-side only -- the decision rule is unchanged).

    Why: after redundancy exclusion n_pairs drops, so condition 3's p-value
    partly measures power, not effect. The report must be able to say
    "magnitude held, restricted subset underpowered" when that is what
    happened -- the same ruler we ask of the reviewers in R1-5 (failure to
    reject != no effect). Rank-biserial r is computed directly from signed
    ranks (version-proof); the CI is a seeded 10k percentile bootstrap of the
    median delta.
    """
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz):
        from scipy.stats import rankdata
        r = rankdata(np.abs(nz))
        pos, neg = float(r[nz > 0].sum()), float(r[nz < 0].sum())
        rank_biserial = (pos - neg) / (pos + neg)
    else:
        rank_biserial = 0.0
    rng = np.random.default_rng(20260820)
    boots = np.median(rng.choice(d, size=(10_000, len(d)), replace=True), axis=1) \
        if len(d) else np.array([np.nan])
    return {"rank_biserial_r": round(float(rank_biserial), 4),
            "median_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                            round(float(np.percentile(boots, 97.5)), 4)]}


def _row_stats(Mrow: np.ndarray, Arow: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rho = float(spearmanr(Mrow, Arow).statistic)
    out = {"rho": round(rho, 4)}
    for k in (3, 5):
        ta = set(np.argsort(-Arow)[:k])
        tm = set(np.argsort(-Mrow)[:k])
        out[f"top{k}_overlap"] = round(len(ta & tm) / k, 4)
    y = np.zeros(len(Arow), dtype=int)
    y[list(np.argsort(-Arow)[:3])] = 1
    out["auroc_top3"] = round(float(roc_auc_score(y, Mrow)), 4) \
        if 0 < y.sum() < len(y) else float("nan")
    return out


def stage_analyze(out_dir: Path = OUT, prior_dir: Path = PRIOR,
                  fiveway_dir: Path = FIVEWAY, datasets=None, seeds=None,
                  redundant_fn=None) -> int:
    """Injectable for the pre-flight selftest (first-author directive,
    2026-08-21): fixtures with known answers exercise every branch before the
    most important computation of the project runs for the first time on real
    matrices (B80's shape: not crashing is not the risk -- silently producing
    wrong numbers is). Defaults reproduce the real analysis unchanged.
    """
    from scipy.stats import wilcoxon

    datasets = datasets or DATASETS
    seeds = seeds or SEEDS
    redundant_fn = redundant_fn or _redundant_set

    def _wilcoxon_p(vec: np.ndarray) -> float:
        # scipy raises on an all-zero delta vector; a zero effect is p = 1,
        # not an exception. Empty -> nan.
        if len(vec) == 0:
            return float("nan")
        if np.allclose(vec, 0.0):
            return 1.0
        return float(wilcoxon(vec).pvalue)

    cells = []
    redundancy = {}
    for ds in datasets:
        R, C = redundant_fn(ds)
        P = None
        for s in seeds:
            tag = f"{ds}_seed{s}_cpu_t{_NT}"
            A = pd.read_csv(out_dir / f"A_{tag}.csv", index_col=0)
            D = pd.read_csv(out_dir / f"D_retrained_{tag}.csv", index_col=0)
            feats = list(D.index)
            if P is None:
                P = pd.read_csv(prior_dir / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                                ).reindex(index=feats, columns=feats)
                n_keep = len([c for c in feats if c not in R])
                # T3.2-R1 doc: not applicable when R is EMPTY or when fewer
                # than 5 non-redundant sources remain per target. The first
                # commit tested only the count; the selftest caught the
                # missing empty-R clause and this now matches the doc.
                redundancy[ds] = {"R": sorted(R), "n_features": len(feats),
                                  "n_keep": n_keep,
                                  "applicable": bool(R) and bool(n_keep - 1 >= 5)}
                C.to_csv(out_dir / f"C_pooled_corr_{ds}.csv")
            mf = {m: pd.read_csv(fiveway_dir / f"D_{ds}_seed{s}_{m}.csv", index_col=0
                                 ).reindex(index=feats, columns=feats)
                  for m in MF_METHODS}
            meta = json.loads((out_dir / f"meta_{tag}.json").read_text())
            for f in A.index:
                srcs = [c for c in feats if c != f]
                scopes = [("full", srcs)]
                kept = [c for c in srcs if c not in R]
                if len(kept) >= 5:               # T3.2-R1 applicability floor
                    scopes.append(("excl_redundant", kept))
                for scope, use in scopes:
                    Arow = A.loc[f, use].to_numpy(float)
                    for name, M in ([("SNI-D", D), ("P", P)]
                                    + [(m, mf[m]) for m in MF_METHODS]):
                        Mrow = M.loc[f, use].to_numpy(float)
                        if np.isnan(Mrow).any():
                            continue
                        cells.append({"dataset": ds, "seed": s, "target": f,
                                      "method": name, "scope": scope,
                                      "n_sources": len(use),
                                      "D_bit_identical": meta["D_bit_identical_to_T2f"],
                                      **_row_stats(Mrow, Arow)})
    df_all = pd.DataFrame(cells)
    df_all.to_csv(out_dir / "faithfulness_cells.csv", index=False)
    df = df_all[df_all.scope == "full"]

    summary = {}
    verdict_inputs = {}
    for ds in datasets:
        g = df[df.dataset == ds]
        blk = {}
        for name in ["SNI-D", "P"] + MF_METHODS:
            gm = g[g.method == name]
            blk[name] = {"rho_median": round(float(gm.rho.median()), 4),
                         "rho_mean": round(float(gm.rho.mean()), 4),
                         "top3_overlap_mean": round(float(gm.top3_overlap.mean()), 4),
                         "auroc_top3_mean": round(float(gm.auroc_top3.mean()), 4),
                         "n": int(len(gm))}
        d_ = g[g.method == "SNI-D"].set_index(["seed", "target"]).rho
        p_ = g[g.method == "P"].set_index(["seed", "target"]).rho
        delta = (d_ - p_).dropna()
        blk["paired_D_minus_P"] = {"n_pairs": int(len(delta)),
                                   "median_delta": round(float(delta.median()), 4),
                                   "mean_delta": round(float(delta.mean()), 4),
                                   "wilcoxon_p": _wilcoxon_p(delta.to_numpy()),
                                   **_paired_effect(delta.to_numpy())}
        summary[ds] = blk
        verdict_inputs[ds] = {"rho_D_median": blk["SNI-D"]["rho_median"],
                              "median_delta": blk["paired_D_minus_P"]["median_delta"],
                              "p": blk["paired_D_minus_P"]["wilcoxon_p"]}

    # Verdict per docs/T32_faithfulness_decision_rule.md (commit 6030500):
    # precedence: unfaithful floor -> D>P in both datasets -> else D~P.
    if any(verdict_inputs[ds]["rho_D_median"] < 0.30 for ds in datasets):
        verdict = "ROW3_D_UNFAITHFUL"
    elif all(verdict_inputs[ds]["median_delta"] > 0
             and verdict_inputs[ds]["p"] < 0.05 for ds in datasets):
        verdict = "ROW1_D_FAITHFUL_P_NOT"
    else:
        verdict = "ROW2_D_EQUIV_P"
    summary["verdict"] = {"rule_commit": "6030500", "verdict": verdict,
                          "inputs": verdict_inputs,
                          "stop_condition_triggered": verdict != "ROW1_D_FAITHFUL_P_NOT"}

    # ---- T3.2-R1 redundancy pre-check (docs/T32_R1_redundancy_precheck.md,
    # committed before any A matrix existed). Both scopes reported regardless
    # of which favors us; the three-condition conjunction below is the
    # prospectively specified operationalization of "advantage holds".
    r1 = {"rule_doc": "docs/T32_R1_redundancy_precheck.md", "threshold": 0.8}
    passes = []
    for ds in datasets:
        ge = df_all[(df_all.dataset == ds) & (df_all.scope == "excl_redundant")]
        blk = dict(redundancy[ds])
        blk["delta_full_median"] = summary[ds]["paired_D_minus_P"]["median_delta"]
        if blk["applicable"] and len(ge):
            d_ = ge[ge.method == "SNI-D"].set_index(["seed", "target"]).rho
            p_ = ge[ge.method == "P"].set_index(["seed", "target"]).rho
            delta = (d_ - p_).dropna()
            blk["excl"] = {
                "n_pairs": int(len(delta)),
                "median_delta": round(float(delta.median()), 4),
                "wilcoxon_p": _wilcoxon_p(delta.to_numpy()),
                **_paired_effect(delta.to_numpy())}
            for name in ["SNI-D", "P"] + MF_METHODS:
                gm = ge[ge.method == name]
                blk[f"rho_median_{name}"] = round(float(gm.rho.median()), 4)
            full = blk["delta_full_median"]
            excl = blk["excl"]["median_delta"]
            # conditions broken out so the report can say WHICH failed --
            # "magnitude held but the restricted subset is underpowered" is a
            # different sentence from "the advantage vanished".
            blk["conditions"] = {
                "c1_sign": bool(excl > 0),
                "c2_magnitude": bool(excl >= 0.5 * full),
                "c3_significance": bool(blk["excl"]["wilcoxon_p"] < 0.05)}
            blk["passes"] = all(blk["conditions"].values())
            passes.append(blk["passes"])
        else:
            blk["excl"] = None
            blk["passes"] = None       # not applicable in this dataset
        r1[ds] = blk
    if not passes:                     # no applicable dataset
        r1["verdict"] = "NOT_APPLICABLE"
    else:                              # carried by the applicable dataset(s)
        r1["verdict"] = "ADVANTAGE_HOLDS" if all(passes) else "SHRUNK_OR_GONE"
    summary["t32_r1_redundancy"] = r1

    (out_dir / "faithfulness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
def stage_r1_sensitivity() -> int:
    """T4.1 -- threshold sensitivity of the T3.2-R1 redundancy check (ESM only).

    The prospectively specified check at 0.8 returned NOT_APPLICABLE (R empty on both
    tables). Reporting only that is incomplete: the threshold choice decides
    applicability itself. Re-run the identical check at 0.5 / 0.6 / 0.7.

    THE VERDICT IS NOT REOPENED. 6030500's ruling stands on the prospectively specified
    0.8; this is a sensitivity analysis describing the conclusion's boundary,
    whichever direction it points (P4 T4.1's explicit declaration).
    """
    rows = []
    for ds in DATASETS:
        C = pd.read_csv(OUT / f"C_pooled_corr_{ds}.csv", index_col=0)
        feats = list(C.index)
        for tau in (0.5, 0.6, 0.7, 0.8):
            R = {f for f in feats if float(C.loc[f].drop(f).max()) > tau}
            n_keep = len(feats) - len(R)
            applicable = bool(R) and (n_keep - 1 >= 5)
            rec = {"dataset": ds, "tau": tau, "R_size": len(R),
                   "n_keep": n_keep, "applicable": applicable,
                   "R": ";".join(sorted(R))}
            if applicable:
                deltas, rhoD, rhoP = [], [], []
                for s in SEEDS:
                    tag = f"{ds}_seed{s}_cpu_t{_NT}"
                    A = pd.read_csv(OUT / f"A_{tag}.csv", index_col=0)
                    D = pd.read_csv(OUT / f"D_retrained_{tag}.csv", index_col=0)
                    P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv",
                                    index_col=0).reindex(index=feats,
                                                         columns=feats)
                    for f in A.index:
                        kept = [c for c in feats if c != f and c not in R]
                        if len(kept) < 5:
                            continue
                        Arow = A.loc[f, kept].to_numpy(float)
                        rd = _row_stats(D.loc[f, kept].to_numpy(float), Arow)["rho"]
                        rp = _row_stats(P.loc[f, kept].to_numpy(float), Arow)["rho"]
                        deltas.append(rd - rp)
                        rhoD.append(rd)
                        rhoP.append(rp)
                dl = np.asarray(deltas, dtype=float)
                from scipy.stats import wilcoxon
                p = (1.0 if np.allclose(dl, 0) else
                     float(wilcoxon(dl).pvalue)) if len(dl) else float("nan")
                rec.update({"n_pairs": len(dl),
                            "rho_D_median": round(float(np.median(rhoD)), 4),
                            "rho_P_median": round(float(np.median(rhoP)), 4),
                            "median_delta": round(float(np.median(dl)), 4),
                            "wilcoxon_p": p, **_paired_effect(dl)})
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r1_threshold_sensitivity.csv", index=False)
    print("VERDICT NOT REOPENED (prospectively specified tau=0.8 stands; ESM "
          "sensitivity only)\n")
    print(df.drop(columns=["R"]).to_string(index=False))
    print("\nR sets:")
    for _, r in df.iterrows():
        if r.R:
            print(f"  {r.dataset} tau={r.tau}: {r.R}")
    return 0


# --------------------------------------------------------------------------- #
# Pre-flight selftest (first-author directive, night of 08-20/21): the analyze
# path must not have its FIRST-EVER execution on the real matrices. Fixtures
# only -- no real A/D/P is read; the prospective specification isolation is untouched.
# --------------------------------------------------------------------------- #
_ZERO_RHO_PERM = [2, 6, 0, 5, 1, 4, 3]   # Spearman(ranks, perm-ranks) == 0 exactly


def _fixture(base: Path, datasets, seeds, d_mode: str, p_mode: str) -> None:
    """Write a full fake input set. Value gaps (0.1) dwarf the seed jitter
    (<=0.003), so ranks -- hence every Spearman -- are exact by construction:
    d_mode='same' gives rho(D,A)=+1, 'zero' gives exactly 0 (via the d^2=56
    permutation); p_mode='reversed' gives rho(P,A)=-1, 'same' gives +1."""
    feats = [f"f{i}" for i in range(8)]
    targets = feats[:6]
    base.mkdir(parents=True, exist_ok=True)
    (base / "prior").mkdir(exist_ok=True)
    (base / "fiveway").mkdir(exist_ok=True)
    for ds in datasets:
        P = pd.DataFrame(0.5, index=feats, columns=feats)
        for s in seeds:
            tag = f"{ds}_seed{s}_cpu_t{_NT}"
            A = pd.DataFrame(np.nan, index=targets, columns=feats)
            D = pd.DataFrame(0.5, index=feats, columns=feats)
            mfM = pd.DataFrame(np.nan, index=feats, columns=feats)
            for f in targets:
                srcs = [c for c in feats if c != f]
                # Strictly descending everywhere: the value map stays monotone,
                # so Spearman identities (+1 / -1 / exactly 0 via the perm)
                # hold exactly. np.roll would break monotonicity and with it
                # the exact-zero construction.
                vals = (np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                        + 0.0004 * s * np.arange(7))
                A.loc[f, srcs] = vals
                D.loc[f, srcs] = (vals if d_mode == "same"
                                  else vals[_ZERO_RHO_PERM])
                P.loc[f, srcs] = (1.0 - vals) if p_mode == "reversed" else vals
                mfM.loc[f, srcs] = vals[::-1]
            A.to_csv(base / f"A_{tag}.csv")
            D.to_csv(base / f"D_retrained_{tag}.csv")
            (base / f"meta_{tag}.json").write_text(json.dumps(
                {"D_bit_identical_to_T2f": True}))
            for m in MF_METHODS:
                # one method left all-NaN to exercise the skip path
                M = pd.DataFrame(np.nan, index=feats, columns=feats) \
                    if m == "SHAP-on-MissForest" else mfM
                M.to_csv(base / "fiveway" / f"D_{ds}_seed{s}_{m}.csv")
        P.to_csv(base / "prior" / f"P_{ds}_seed1_cpu_t2.csv")


def _run_case(tmp: Path, name: str, d_mode: str, p_mode: str, R_by_ds: dict):
    import contextlib
    import io
    ds_list = list(R_by_ds)
    seeds = [1, 2, 3]
    base = tmp / name
    _fixture(base, ds_list, seeds, d_mode, p_mode)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stage_analyze(out_dir=base, prior_dir=base / "prior",
                      fiveway_dir=base / "fiveway", datasets=ds_list,
                      seeds=seeds, redundant_fn=lambda ds: (R_by_ds[ds], pd.DataFrame()))
    return json.loads((base / "faithfulness_summary.json").read_text())


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory(prefix="t32_selftest_") as td:
        tmp = Path(td)
        Rgood = {"FA": {"f6", "f7"}, "FB": {"f6", "f7"}}   # kept>=5, non-empty

        s1 = _run_case(tmp, "s1", "same", "reversed", Rgood)
        v, r1 = s1["verdict"], s1["t32_r1_redundancy"]
        check(v["verdict"] == "ROW1_D_FAITHFUL_P_NOT", f"S1 verdict ROW1 (got {v['verdict']})")
        check(s1["FA"]["SNI-D"]["rho_median"] == 1.0
              and s1["FA"]["P"]["rho_median"] == -1.0, "S1 rho(D,A)=+1, rho(P,A)=-1 exact")
        pd_ = s1["FA"]["paired_D_minus_P"]
        check(pd_["median_delta"] == 2.0 and pd_["n_pairs"] == 18, "S1 delta=2.0, n=18")
        check(pd_["rank_biserial_r"] == 1.0 and pd_["median_ci95"] == [2.0, 2.0],
              "S1 effect r=+1, CI [2,2]")
        check(r1["verdict"] == "ADVANTAGE_HOLDS"
              and r1["FA"]["conditions"] == {"c1_sign": True, "c2_magnitude": True,
                                             "c3_significance": True},
              "S1 R1 ADVANTAGE_HOLDS, all three conditions True")
        check(r1["FA"]["excl"]["n_pairs"] == 18 and r1["FA"]["n_keep"] == 6,
              "S1 excl n_pairs reported alongside full")

        s2 = _run_case(tmp, "s2", "zero", "same", Rgood)
        check(s2["verdict"]["verdict"] == "ROW3_D_UNFAITHFUL",
              f"S2 verdict ROW3 (rho_D=0 < 0.30 floor; got {s2['verdict']['verdict']})")
        check(abs(s2["FA"]["SNI-D"]["rho_median"]) < 1e-9, "S2 rho(D,A)=0 exact")
        check(s2["t32_r1_redundancy"]["verdict"] == "SHRUNK_OR_GONE",
              "S2 R1 reported under ROW3 too (SHRUNK_OR_GONE)")

        s3 = _run_case(tmp, "s3", "same", "same", Rgood)
        check(s3["verdict"]["verdict"] == "ROW2_D_EQUIV_P",
              f"S3 verdict ROW2 on D==P (got {s3['verdict']['verdict']})")
        check(s3["FA"]["paired_D_minus_P"]["wilcoxon_p"] == 1.0,
              "S3 all-zero deltas -> guarded p=1.0, no crash")

        s4 = _run_case(tmp, "s4", "same", "reversed",
                       {"FA": {"f2", "f3", "f4", "f5", "f6", "f7"}, "FB": set()})
        r14 = s4["t32_r1_redundancy"]
        check(r14["verdict"] == "NOT_APPLICABLE"
              and r14["FA"]["excl"] is None and r14["FB"]["excl"] is None,
              "S4 kept<5 and empty-R both inapplicable -> NOT_APPLICABLE, no crash")
        check(s4["verdict"]["verdict"] == "ROW1_D_FAITHFUL_P_NOT",
              "S4 main verdict unaffected by R1 inapplicability")

        cols = set(pd.read_csv(tmp / "s1" / "faithfulness_cells.csv").columns)
        need = {"dataset", "seed", "target", "method", "scope", "n_sources",
                "D_bit_identical", "rho", "top3_overlap", "top5_overlap",
                "auroc_top3"}
        check(need.issubset(cols), f"cells columns complete (missing: {need - cols})")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "analyze", "selftest", "r1-sensitivity"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--variant", choices=["SNI", "NoPrior"], default="SNI",
                    help="T4.3: NoPrior retrains with alpha=0 (tags NP_*)")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "r1-sensitivity":
        return stage_r1_sensitivity()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        for ds in a.datasets:
            for s in a.seeds:
                run_one(ds, s, variant=a.variant)
        return 0
    return stage_analyze()


if __name__ == "__main__":
    raise SystemExit(main())
