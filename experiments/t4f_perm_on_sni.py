"""T4F -- Permutation-on-SNI: the host-controlled comparator.

Presentation prospectively specified at docs/T4F_presentation_rule.md (f224cd2) BEFORE
any number here existed. Three stages:

  --stage train   retrain the pilot's 15 synthetic hosts (models were never
                  stored), verify each reproduces the pilot's stored SNI-D
                  matrix, then compute the permutation-ablation matrix with
                  T3.2's exact recipe (standardized-column row permutation,
                  missingness untouched, 5 permutations averaged; continuous
                  targets: NRMSE-std increment; categorical targets:
                  Macro-F1 decrement). Cached per cell (B79).
  --stage score   six-way scoring on the pilot's scorer and common row set.
  --stage real    (a) cross-seed stability of the REAL-table ablation
                  matrices T3.2 already computed (A_{ds}_seed*.csv) --
                  pairwise Spearman over target-row entries + top-3 Jaccard;
                  (b) audit-cost timing: reload the stored T3.2 state_dicts
                  and time the full ablation loop once per dataset.

    env PYTHONHASHSEED=2025 python experiments/t4f_perm_on_sni.py --stage train --regimes linear_gaussian
    env PYTHONHASHSEED=2025 python experiments/t4f_perm_on_sni.py --stage score
    env PYTHONHASHSEED=2025 python experiments/t4f_perm_on_sni.py --stage real
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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

PILOT = CODE_ROOT / "results" / "T2.5_pilot"
FAITH = CODE_ROOT / "results" / "T3_faithfulness"
OUT = CODE_ROOT / "results" / "T4_perm_on_sni"

REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SYNTH_SEEDS = [2025, 2026, 2027, 2028, 2029]
REAL_SEEDS = [1, 2, 3, 5, 8]
DATASETS = ["MIMIC", "eICU"]
N_PERM = 5
PILOT_METHODS = ["SNI-D", "MissForest-importance", "SHAP-on-MissForest",
                 "Permutation-on-MissForest"]


def _ablate(imp, X_final, mask_df, complete, feats, seed, cat_vars):
    """T3.2's recipe verbatim, plus the categorical branch (Macro-F1 drop).

    RNG registration matches faithfulness.py:
    default_rng(10_000*seed + 100*f_index + 5*j_index + r).
    """
    import torch
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    targets = [f for f in feats if int(mask_df[f].sum()) > 0]
    A = pd.DataFrame(np.nan, index=targets, columns=feats)
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
        model = imp.models[f]
        model.eval()
        is_cat = f in cat_vars

        if is_cat:
            truth = complete[f].astype(str).to_numpy()[miss]
            le = imp.encoders[f]

            def _err(Zmat):
                with torch.no_grad():
                    logits = model(torch.tensor(Zmat[miss],
                                                dtype=torch.float32))[0]
                lab = le.inverse_transform(
                    logits.argmax(dim=1).cpu().numpy()).astype(str)
                return 1.0 - f1_score(truth, lab, average="macro")
        else:
            from sklearn.preprocessing import StandardScaler as _S
            y_obs = pd.to_numeric(X_final[f], errors="coerce"
                                  ).to_numpy(float)[present]
            scaler_y = _S().fit(y_obs.reshape(-1, 1))
            truth = pd.to_numeric(complete[f], errors="coerce"
                                  ).to_numpy(float)[miss]
            sd = float(truth.std()) or 1.0

            def _err(Zmat):
                with torch.no_grad():
                    yh = model(torch.tensor(Zmat[miss],
                                            dtype=torch.float32))[0]
                pred = scaler_y.inverse_transform(
                    yh.cpu().numpy().reshape(-1, 1)).flatten()
                return float(np.sqrt(np.mean((truth - pred) ** 2)) / sd)

        e0 = _err(Z_s)
        for jj, j in enumerate(srcs):
            deltas = []
            for r in range(N_PERM):
                rng = np.random.default_rng(10_000 * seed + 100 * fi
                                            + 5 * jj + r)
                Zp = Z_s.copy()
                Zp[:, jj] = Zp[rng.permutation(n), jj]
                deltas.append(_err(Zp) - e0)
            A.loc[f, j] = float(np.mean(deltas))
    return A


def run_synth(regime: str, seed: int) -> None:
    import yaml
    from common import determinism
    from pilot_r21 import load_cell, row_normalise
    from sni.imputer import SNIConfig, SNIImputer

    tag = f"{regime}_s{seed}"
    if (OUT / f"PERM_{tag}.csv").exists() and (OUT / f"D_RETR_{tag}.csv").exists():
        print(f"[cached] {tag}")
        return
    complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
    cols = list(complete.columns)

    proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                            ).read_text())["protocol"]
    determinism.apply("deterministic", seed=seed)
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.epochs = int(proto["epochs"]["SNI"])
    imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
    t0 = time.time()
    X_final = imp.impute(X_missing=missing[imp.all_vars], X_complete=None,
                         mask_df=mask[imp.all_vars])
    wall = time.time() - t0

    D = row_normalise(imp.compute_dependency_matrix().reindex(
        index=cols, columns=cols).fillna(0.0))
    # P4-G section 0.2: keep the retrained host's OWN D so the host-controlled
    # contrast (retrained D vs retrained Perm) is bitwise same-host, while
    # Table 3's SNI-D row keeps the archived pilot caliber.
    D.to_csv(OUT / f"D_RETR_{tag}.csv")
    D_ref = pd.read_csv(PILOT / f"D_{regime}_s{seed}_SNI-D.csv", index_col=0
                        ).reindex(index=cols, columns=cols)
    dmax = float(np.abs(D.to_numpy() - D_ref.to_numpy()).max())
    if dmax != 0.0:
        print(f"WARNING: retrained D differs from pilot store "
              f"(max {dmax:.3e}) for {tag}", file=sys.stderr)

    mask_df = mask[imp.all_vars].astype(bool)
    t1 = time.time()
    A = _ablate(imp, X_final, mask_df, complete, imp.all_vars, seed, set(cat))
    audit = time.time() - t1
    A = A.reindex(index=cols, columns=cols)
    A.to_csv(OUT / f"PERM_{tag}.csv")
    (OUT / f"meta_{tag}.json").write_text(json.dumps(
        {"tag": tag, "wall_train_sec": round(wall, 1),
         "audit_sec": round(audit, 1), "D_max_abs_diff": dmax,
         "n_targets": int(A.notna().any(axis=1).sum())}, indent=2))
    print(f"[ok] {tag} train={wall:.0f}s audit={audit:.0f}s "
          f"D_bitwise={dmax == 0.0}", flush=True)


def stage_score() -> int:
    from pilot_r21 import load_cell, measured_rows, score
    rows = []
    for regime in REGIMES:
        for s in SYNTH_SEEDS:
            complete, missing, mask, G, cat, cont, _ = load_cell(regime, s)
            cols = list(complete.columns)
            mats = {m: pd.read_csv(PILOT / f"D_{regime}_s{s}_{m}.csv",
                                   index_col=0).reindex(index=cols,
                                                        columns=cols).fillna(0.0)
                    for m in PILOT_METHODS}
            mats["Permutation-on-SNI"] = pd.read_csv(
                OUT / f"PERM_{regime}_s{s}.csv", index_col=0).reindex(
                index=cols, columns=cols).fillna(0.0)
            # P4-G 0.2: the retrained host's own D, so the host-controlled
            # contrast (retrained D vs retrained Perm) is bitwise same-host.
            # Table 3's SNI-D row stays on the archived pilot caliber; this
            # object quantifies the archived-vs-retrained score gap (B84:
            # the pilot host ran with unpinned BLAS threads).
            mats["SNI-D-retrained"] = pd.read_csv(
                OUT / f"D_RETR_{regime}_s{s}.csv", index_col=0).reindex(
                index=cols, columns=cols).fillna(0.0)
            tap = pd.read_csv(CODE_ROOT / "results" / "T2g_prior_attribution"
                              / f"P_synth_{regime}_s{s}.csv", index_col=0
                              ).reindex(index=cols, columns=cols).fillna(0.0)
            mats["P-alone"] = tap
            common = np.ones(len(cols), dtype=bool)
            for M in mats.values():
                common &= measured_rows(M)
            for name, M in mats.items():
                sc = score(M, G, keep=common)
                rows.append({"regime": regime, "seed": s, "method": name,
                             "n_rows_common": int(common.sum()), **sc})
            pd.DataFrame(rows).to_csv(OUT / "t4f_sixway_cells.csv",
                                      index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("method")[["auroc", "auprc", "prec_at_k", "shd"]]
          .mean().round(4).to_string())

    # ---- prospectively specified verdict (docs/T4F_score_verdict_rule.md, 0cf4662,
    # committed before any comparison number existed). Mechanical: the code
    # reads the rule, not the operator reading the numbers.
    from scipy.stats import rankdata, wilcoxon
    piv = df.pivot_table(index=["regime", "seed"], columns="method",
                         values="auroc")

    def _paired(colA: str, colB: str) -> dict:
        d = (piv[colA] - piv[colB]).dropna()
        vec = d.to_numpy(float)
        med = float(np.median(vec))
        pval = (1.0 if np.allclose(vec, 0) else
                float(wilcoxon(vec).pvalue)) if len(vec) else float("nan")
        nz = vec[vec != 0]
        if len(nz):
            r = rankdata(np.abs(nz))
            pos, neg = float(r[nz > 0].sum()), float(r[nz < 0].sum())
            rb = (pos - neg) / (pos + neg)
        else:
            rb = 0.0
        rng = np.random.default_rng(20260822)
        meds = []
        for _ in range(2000):
            bs = rng.choice(SYNTH_SEEDS, size=len(SYNTH_SEEDS), replace=True)
            vals = np.concatenate([d.xs(sd, level="seed").to_numpy(float)
                                   for sd in bs])
            meds.append(np.median(vals))
        per_regime = {rg: float(np.median(
            d.xs(rg, level="regime").to_numpy(float))) for rg in REGIMES}
        return {"n_pairs": int(len(vec)), "median_delta": round(med, 4),
                "wilcoxon_p": round(pval, 5), "rank_biserial_r": round(rb, 4),
                "median_ci95_seedboot": [round(float(np.percentile(meds, 2.5)), 4),
                                         round(float(np.percentile(meds, 97.5)), 4)],
                "per_regime_median": {k: round(v, 4)
                                      for k, v in per_regime.items()}}

    primary = _paired("Permutation-on-SNI", "SNI-D-retrained")
    n_dir = sum(v > 0 for v in primary["per_regime_median"].values())
    wins = (primary["median_delta"] > 0 and primary["wilcoxon_p"] < 0.05
            and n_dir >= 2)
    verdict = {
        "rule_commit": "0cf4662",
        "primary_pair": "Permutation-on-SNI vs SNI-D-retrained (same host, bitwise)",
        "primary": primary,
        "conditions": {"c1_median_pos": bool(primary["median_delta"] > 0),
                       "c2_wilcoxon": bool(primary["wilcoxon_p"] < 0.05),
                       "c3_regimes_2of3": bool(n_dir >= 2),
                       "n_regimes_direction": n_dir},
        "verdict": "SAME_HOST_POSTHOC_WINS" if wins else "INDISTINGUISHABLE",
        "continuity_archived_pair": _paired("Permutation-on-SNI", "SNI-D"),
    }
    (OUT / "t4f_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def _load_host(ds: str, seed: int, prefix: str = ""):
    """Rebuild the T3.2 host (continuous targets only) from stored artifacts.

    `prefix` selects the arm: "" is the with-prior host, "NP_" the no-prior
    control's. One loader for both, so the no-prior band cannot drift into a
    second implementation of the same load (T6.1 addendum 2026-08-29d SS3).
    """
    import torch
    from baselines.schema import DataSchema
    from sni.cpfa import EnhancedCPFA
    from sni.imputer import SNIConfig, SNIImputer

    tag = f"{prefix}{ds}_seed{seed}_cpu_t2"
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    cat = list(schema.categorical_vars)
    cont = list(schema.continuous_vars)
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=seed, use_gpu=False))
    X_final = pd.read_csv(FAITH / f"Xfinal_{tag}.csv")
    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[imp.all_vars]
    states = torch.load(FAITH / f"models_{tag}.pt", map_location="cpu")
    targets = [f for f in imp.all_vars if int(mask_df[f].sum()) > 0]
    for f, sd in states.items():
        if f not in targets:
            continue
        cat_idx = imp._get_categorical_indices_excluding_target(f, imp.all_vars)
        m = EnhancedCPFA(input_dim=len(imp.all_vars) - 1,
                         emb_dim=imp.cfg.emb_dim,
                         num_heads=imp.cfg.num_heads,
                         hidden_dims=list(imp.cfg.hidden_dims),
                         output_dim=1, is_classification=False,
                         cat_indices=cat_idx,
                         use_cat_embedding=imp.cfg.use_cat_embedding,
                         use_multiscale=False, mask_aware=False)
        m.load_state_dict(sd)
        imp.models[f] = m
    return imp, X_final, mask_df, complete, set(cat)


def stage_within(ds_list=None) -> int:
    """T4G.2 -- within-seed consistency of A under fresh permutation draws.

    The plug for the section-1.1 argument's hole: rho_A(cross-seed) ~ 0.5
    could be host variation (the claim) or 5-permutation sampling noise (the
    refutation). Two fresh draw-groups per seed on the SAME stored host;
    within-seed pairs {orig, g1, g2} vs the cross-seed pairs, same rows, same
    caliber. Branch rule fixed in the P4-G instrument BEFORE these numbers:
    within >= 0.85 and cross ~ 0.5 => host variation, argument stands;
    within ~ 0.5 => measurement noise, the argument is NOT written.
    """
    from scipy.stats import spearmanr

    rows = []
    for ds in (ds_list or DATASETS):
        for seed in REAL_SEEDS:
            tag = f"{ds}_seed{seed}_cpu_t2"
            groups = {0: pd.read_csv(FAITH / f"A_{tag}.csv", index_col=0)}
            need = [g for g in (1, 2)
                    if not (OUT / f"A_G{g}_{tag}.csv").exists()]
            if need:
                imp, X_final, mask_df, complete, cats = _load_host(ds, seed)
                for g in need:
                    Ag = _ablate(imp, X_final, mask_df, complete,
                                 imp.all_vars, 900_000_000 + 1_000_000 * g
                                 + seed, cats)
                    Ag.to_csv(OUT / f"A_G{g}_{tag}.csv")
            for g in (1, 2):
                groups[g] = pd.read_csv(OUT / f"A_G{g}_{tag}.csv", index_col=0)
            first = groups[0]
            sel = first.notna().to_numpy()
            for a, b in combinations(sorted(groups), 2):
                A = groups[a].to_numpy(float)[sel]
                B = groups[b].to_numpy(float)[sel]
                r = spearmanr(A, B)
                rows.append({"dataset": ds, "seed": seed, "a": a, "b": b,
                             "spearman": round(float(r.statistic), 4)})
            pd.DataFrame(rows).to_csv(OUT / "within_seed_consistency.csv",
                                      index=False)
            print(f"[ok] within {tag}: "
                  + ", ".join(f"{r['a']}-{r['b']}={r['spearman']:.3f}"
                              for r in rows[-3:]), flush=True)
    df = pd.DataFrame(rows)
    cross = pd.read_csv(OUT / "perm_on_sni_real_stability.csv")
    for ds in (ds_list or DATASETS):
        w = df[df.dataset == ds].spearman
        c = cross[cross.dataset == ds].spearman
        print(f"{ds}: WITHIN-seed rho mean {w.mean():.4f} (min {w.min():.4f}, "
              f"n={len(w)})  vs CROSS-seed mean {c.mean():.4f}")
    return 0


def stage_real() -> int:
    from scipy.stats import spearmanr

    # (a) cross-seed stability of the REAL-table ablation matrices (T3.2's A)
    stab_rows = []
    for ds in DATASETS:
        mats = {s: pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv",
                               index_col=0) for s in REAL_SEEDS}
        first = mats[REAL_SEEDS[0]]
        sel = first.notna().to_numpy()
        for a, b in combinations(REAL_SEEDS, 2):
            A = mats[a].to_numpy(float)[sel]
            B = mats[b].to_numpy(float)[sel]
            r = spearmanr(A, B)
            ja = []
            for i, f in enumerate(first.index):
                ra = mats[a].loc[f].dropna()
                rb = mats[b].loc[f].dropna()
                ta = set(ra.sort_values(ascending=False).index[:3])
                tb = set(rb.sort_values(ascending=False).index[:3])
                ja.append(len(ta & tb) / len(ta | tb))
            stab_rows.append({"dataset": ds, "a": a, "b": b,
                              "spearman": round(float(r.statistic), 4),
                              "p": float(r.pvalue),
                              "top3_jaccard": round(float(np.mean(ja)), 4)})
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT / "perm_on_sni_real_stability.csv", index=False)

    # agreement with TAP, per seed (feeds the six-object stability table)
    agree_rows = []
    for ds in DATASETS:
        P = pd.read_csv(CODE_ROOT / "results" / "T2g_prior_attribution"
                        / f"P_{ds}_seed1_cpu_t2.csv", index_col=0)
        for s in REAL_SEEDS:
            A = pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t2.csv", index_col=0)
            sel = A.notna()
            av, pv = [], []
            for f in A.index:
                cols_f = list(A.columns[sel.loc[f]])
                av.extend(A.loc[f, cols_f].to_numpy(float))
                pv.extend(P.loc[f, cols_f].to_numpy(float))
            r = spearmanr(av, pv)
            agree_rows.append({"dataset": ds, "seed": s,
                               "rho_with_P": round(float(r.statistic), 4)})
    pd.DataFrame(agree_rows).to_csv(OUT / "perm_on_sni_agreement_with_P.csv",
                                    index=False)
    for ds in DATASETS:
        g = stab[stab.dataset == ds]
        print(f"{ds}: perm-on-SNI cross-seed rho mean {g.spearman.mean():.4f} "
              f"min {g.spearman.min():.4f} top3J {g.top3_jaccard.mean():.4f}")

    # (b) audit-cost timing from the stored T3.2 state_dicts, one seed per ds
    import torch
    import yaml
    from baselines.schema import DataSchema
    from sni.cpfa import EnhancedCPFA
    from sni.imputer import SNIConfig, SNIImputer

    cost_rows = []
    for ds in DATASETS:
        tag = f"{ds}_seed1_cpu_t2"
        schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
        cat = list(schema.categorical_vars)
        cont = list(schema.continuous_vars)
        imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                         config=SNIConfig(seed=1, use_gpu=False))
        X_final = pd.read_csv(FAITH / f"Xfinal_{tag}.csv")
        complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                               / f"{ds}_complete.csv")
        mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                       / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
        mask_df = pd.DataFrame(mask, columns=list(complete.columns)
                               )[imp.all_vars]
        states = torch.load(FAITH / f"models_{tag}.pt", map_location="cpu")
        targets = [f for f in imp.all_vars if int(mask_df[f].sum()) > 0]
        for f, sd in states.items():
            if f not in targets:
                # always-observed features (the two categoricals) were trained
                # with the multiscale classification architecture; the ablation
                # never queries them as targets, so they are not reloaded.
                continue
            cat_idx = imp._get_categorical_indices_excluding_target(
                f, imp.all_vars)
            m = EnhancedCPFA(input_dim=len(imp.all_vars) - 1,
                             emb_dim=imp.cfg.emb_dim,
                             num_heads=imp.cfg.num_heads,
                             hidden_dims=list(imp.cfg.hidden_dims),
                             output_dim=1, is_classification=False,
                             cat_indices=cat_idx,
                             use_cat_embedding=imp.cfg.use_cat_embedding,
                             use_multiscale=False, mask_aware=False)
            m.load_state_dict(sd)
            imp.models[f] = m
        t0 = time.time()
        _ = _ablate(imp, X_final, mask_df, complete, imp.all_vars, 1, set(cat))
        audit = time.time() - t0
        cost_rows.append({"dataset": ds, "method": "Permutation-on-SNI",
                          "audit_sec": round(audit, 1)})
        print(f"{ds}: perm-on-SNI audit wall {audit:.1f}s")
    pd.DataFrame(cost_rows).to_csv(OUT / "perm_on_sni_audit_cost.csv",
                                   index=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "score", "real", "within"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--seeds", type=int, nargs="*", default=SYNTH_SEEDS)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        for regime in a.regimes:
            for s in a.seeds:
                run_synth(regime, s)
        return 0
    if a.stage == "score":
        return stage_score()
    if a.stage == "within":
        return stage_within(a.datasets)
    return stage_real()


if __name__ == "__main__":
    raise SystemExit(main())
