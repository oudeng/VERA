"""T5.2 probe 2 -- joint group permutation on archived models (rules:
docs/T52_probe_triangulation_rules.md, commit d5796c3, committed before
any group-permutation measurement existed).

Zero retraining: models are cold-loaded from the archived models_{tag}.pt
(architecture flags recovered from the state dict, strict load), the
encoder is stateless, and the scalers are deterministic refits on the
archived Xfinal + the frozen mask -- verified per host by recomputing one
probe-1 cell (same RNG streams) and asserting bit-level agreement with
the stored A matrix.

Measurement (rules SS3): clusters = connected components of the
prospectively specified pooled correlation at tau (headline 0.8; sensitivity 0.6,
0.7); B[f, G] = mean over 5 reps of the pooled std-normalized error
delta when ALL member columns of G are permuted by ONE shared row
permutation; RNG = default_rng(70_000 + 1_000*seed + 100*f_index +
20*g_index + r). Object group scores = member MEAN (member sum as
sensitivity). Per-cell rho_group = Spearman(B[f, .], M_group[f, .]);
Delta rho_group (D - TAP) feeds the T5.1 machinery per (table, variant,
tau).

    env PYTHONHASHSEED=2025 python experiments/t52_probe2.py --stage selftest
    env PYTHONHASHSEED=2025 python experiments/t52_probe2.py --stage run [--hosts SNI NoPrior]
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
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
OUT = CODE_ROOT / "results" / "T5_probe2"
DATASETS = ["MIMIC", "eICU"]
TAUS = (0.8, 0.6, 0.7)          # 0.8 = prospectively specified headline
N_REP = 5


# ---------------- pure pieces (selftested) --------------------------------- #
def partition(C: pd.DataFrame, tau: float) -> list:
    """Connected components of |corr| > tau, singletons included; sorted
    by first-member column order (rules SS3.1)."""
    feats = list(C.index)
    idx = {f: i for i, f in enumerate(feats)}
    parent = list(range(len(feats)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, f in enumerate(feats):
        for j, g in enumerate(feats):
            if j <= i:
                continue
            if float(C.loc[f, g]) > tau:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    groups: dict = {}
    for i, f in enumerate(feats):
        groups.setdefault(find(i), []).append(f)
    comps = [sorted(v, key=lambda x: idx[x]) for v in groups.values()]
    return sorted(comps, key=lambda g: idx[g[0]])


def group_scores(M_row: pd.Series, grps: list, how: str = "mean") -> list:
    fn = np.mean if how == "mean" else np.sum
    return [float(fn([float(M_row[c]) for c in g])) for g in grps]


def rho_cells(B_vals: list, M_vals: list) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(B_vals), np.asarray(M_vals)).statistic)


# ---------------- cold-load ------------------------------------------------- #
def load_host(ds: str, seed: int, variant: str):
    """Rebuild the imputer shell + per-target models from the archive;
    returns (imp, models{f}, X_final, mask_df, complete, feats)."""
    import torch
    from baselines.schema import DataSchema
    from sni.cpfa import EnhancedCPFA
    from sni.imputer import SNIConfig, SNIImputer

    tag = (f"{ds}_seed{seed}_cpu_t{_NT}" if variant == "SNI"
           else f"NP_{ds}_seed{seed}_cpu_t{_NT}")
    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    X_final = pd.read_csv(FAITH / f"Xfinal_{tag}.csv")
    imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                     continuous_vars=list(schema.continuous_vars),
                     config=SNIConfig(seed=seed, use_gpu=False))
    states = torch.load(FAITH / f"models_{tag}.pt", map_location="cpu",
                        weights_only=True)
    # The probe's targets (the A-matrix rows) are the masked columns, all
    # continuous on both tables -- reconstructed exactly as the imputer's
    # continuous branch builds them (imputer.py:813). Categorical models
    # (DualPathCPFA) are never probed and are not reconstructed.
    import re
    targets = [f for f in feats
               if f in schema.continuous_vars
               and int(mask_df[f].sum()) > 0]
    models = {}
    for f in targets:
        sd = states[f]
        mlp_ws = {int(re.match(r"mlp\.(\d+)\.weight", k).group(1)): v
                  for k, v in sd.items()
                  if re.match(r"mlp\.(\d+)\.weight", k)}
        out_dim = int(mlp_ws[max(mlp_ws)].shape[0])
        assert out_dim == 1 and not any(
            k.startswith("continuous_path") for k in sd), \
            f"{tag}/{f}: not a continuous-branch model"
        m = EnhancedCPFA(
            input_dim=len(feats) - 1,
            emb_dim=imp.cfg.emb_dim,
            num_heads=imp.cfg.num_heads,
            hidden_dims=list(imp.cfg.hidden_dims),
            output_dim=1,
            is_classification=False,
            cat_indices=imp._get_categorical_indices_excluding_target(
                f, imp.all_vars),
            use_cat_embedding=imp.cfg.use_cat_embedding,
            use_multiscale=False,
            mask_aware=bool(any(k.startswith("missing_embedding")
                                for k in sd)),
        )
        m.load_state_dict(sd, strict=True)
        m.eval()
        models[f] = m
    return imp, models, X_final, mask_df, complete, feats


def _target_ctx(imp, models, X_final, mask_df, complete, feats, f):
    """faithfulness.run_one's inference pieces for one target, verbatim
    caliber."""
    import torch
    from sklearn.preprocessing import StandardScaler
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
    model = models[f]

    def predict(Zmat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            yh = model(torch.tensor(Zmat[miss], dtype=torch.float32))[0]
        return scaler_y.inverse_transform(
            yh.cpu().numpy().reshape(-1, 1)).flatten()

    return Z_s, srcs, miss, truth, sd, predict


def verify_reload(imp, models, X_final, mask_df, complete, feats,
                  seed: int, A_stored: pd.DataFrame) -> float:
    """Recompute ONE stored probe-1 cell (first target, first source)
    with the original RNG streams; return |diff| vs the archived A."""
    f = list(A_stored.index)[0]
    fi = feats.index(f)
    Z_s, srcs, miss, truth, sd, predict = _target_ctx(
        imp, models, X_final, mask_df, complete, feats, f)
    e0 = float(np.sqrt(np.mean((truth - predict(Z_s)) ** 2))) / sd
    jj, j = 0, srcs[0]
    n = len(Z_s)
    deltas = []
    for r in range(N_REP):
        rng = np.random.default_rng(10_000 * seed + 100 * fi + 5 * jj + r)
        Zp = Z_s.copy()
        Zp[:, jj] = Zp[rng.permutation(n), jj]
        e = float(np.sqrt(np.mean((truth - predict(Zp)) ** 2))) / sd
        deltas.append(e - e0)
    return abs(float(np.mean(deltas)) - float(A_stored.loc[f, j]))


# ---------------- measurement ---------------------------------------------- #
def measure_host(ds: str, seed: int, variant: str, parts: dict,
                 n_rep: int = N_REP) -> dict:
    """B[f, G] for every tau partition; returns
    {tau: {f: {gidx: delta}}} plus the reload-verification deviation."""
    imp, models, X_final, mask_df, complete, feats = load_host(
        ds, seed, variant)
    tag = ("" if variant == "SNI" else "NP_") + f"{ds}_seed{seed}_cpu_t{_NT}"
    A_stored = pd.read_csv(FAITH / f"A_{tag}.csv", index_col=0)
    dev = verify_reload(imp, models, X_final, mask_df, complete, feats,
                        seed, A_stored)
    assert dev <= 1e-9, f"cold-load verification failed {tag}: dev={dev:.2e}"
    out = {}
    for tau, grps in parts.items():
        per_f = {}
        for f in A_stored.index:
            fi = feats.index(f)
            Z_s, srcs, miss, truth, sd, predict = _target_ctx(
                imp, models, X_final, mask_df, complete, feats, f)
            e0 = float(np.sqrt(np.mean((truth - predict(Z_s)) ** 2))) / sd
            n = len(Z_s)
            cell = {}
            for gidx, g in enumerate(grps):
                members = [c for c in g if c != f]
                if not members:
                    continue                     # rules SS3.3
                cols = [srcs.index(c) for c in members]
                deltas = []
                for r in range(n_rep):
                    rng = np.random.default_rng(
                        70_000 + 1_000 * seed + 100 * fi + 20 * gidx + r)
                    perm = rng.permutation(n)
                    Zp = Z_s.copy()
                    for cj in cols:
                        Zp[:, cj] = Zp[perm, cj]
                    e = float(np.sqrt(np.mean(
                        (truth - predict(Zp)) ** 2))) / sd
                    deltas.append(e - e0)
                cell[gidx] = float(np.mean(deltas))
            per_f[f] = cell
        out[tau] = per_f
    return {"cells": out, "reload_dev": dev}


def stage_run(hosts: list) -> int:
    from t51_cluster_stats import (EXP_SEEDS, SEEDS, aggregates,
                                   hier_boot_ci, sign_flip_exact)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    summary: dict = {"rules": "docs/T52_probe_triangulation_rules.md "
                              "@ d5796c3", "families": {}}
    for ds in DATASETS:
        C = pd.read_csv(FAITH / f"C_pooled_corr_{ds}.csv", index_col=0)
        parts = {tau: partition(C, tau) for tau in TAUS}
        feats = list(C.index)
        P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                        ).reindex(index=feats, columns=feats)
        for variant in hosts:
            seeds = SEEDS + EXP_SEEDS
            blocks: dict = {tau: {} for tau in TAUS}
            for s in seeds:
                pre = "" if variant == "SNI" else "NP_"
                tagf = f"{pre}{ds}_seed{s}_cpu_t{_NT}"
                if not (FAITH / f"models_{tagf}.pt").exists():
                    continue
                m = measure_host(ds, s, variant, parts)
                D = pd.read_csv(FAITH / f"D_retrained_{tagf}.csv",
                                index_col=0)
                for tau in TAUS:
                    grps = parts[tau]
                    dl = []
                    for f, cell in m["cells"][tau].items():
                        gidx = sorted(cell)
                        B = [cell[i] for i in gidx]
                        gd = group_scores(
                            D.loc[f],
                            [[c for c in grps[i] if c != f] for i in gidx])
                        gp = group_scores(
                            P.loc[f],
                            [[c for c in grps[i] if c != f] for i in gidx])
                        rD, rP = rho_cells(B, gd), rho_cells(B, gp)
                        rows.append({"dataset": ds, "variant": variant,
                                     "seed": s, "target": f, "tau": tau,
                                     "n_groups": len(gidx),
                                     "rho_group_D": round(rD, 6),
                                     "rho_group_TAP": round(rP, 6),
                                     "reload_dev": m["reload_dev"]})
                        dl.append(rD - rP)
                    blocks[tau][s] = np.array(dl)
                print(f"[host ok] {variant} {ds} seed {s} "
                      f"(reload dev {m['reload_dev']:.1e})", flush=True)
            for tau in TAUS:
                if not blocks[tau]:
                    continue
                key = f"{variant}|{ds}|tau={tau}"
                summary["families"][key] = {
                    "n_seeds": len(blocks[tau]),
                    "m1": sign_flip_exact(blocks[tau]),
                    "m2_ci95": hier_boot_ci(blocks[tau]),
                    "aggregates": aggregates(blocks[tau])}
    pd.DataFrame(rows).to_csv(OUT / "probe2_cells.csv", index=False)
    (OUT / "probe2_summary.json").write_text(json.dumps(summary, indent=1))
    for k, f in summary["families"].items():
        print(f"{k:24s} eff={f['m1']['effect_pooled_median']:+.4f} "
              f"p2={f['m1']['p_two_sided']:.6f} "
              f"CI[{f['m2_ci95'][0]:+.3f},{f['m2_ci95'][1]:+.3f}]")
    print(f"[ok] wrote {OUT}/probe2_cells.csv + probe2_summary.json")
    return 0


def stage_qualifiers() -> int:
    """The rules' two remaining reliability qualifiers (SS 'Triangulated
    reading'): (a) permutation-count sensitivity -- 5 vs 10 repetitions on
    one seed per table (fresh streams for r in 5..9 extend the same
    prospectively specified formula); (b) probe-1/probe-2 agreement -- at tau=0.8
    the partition is all singletons on both tables, so B[f, {c}] measures
    the same quantity as probe-1's A[f, c] with fresh draws: rho-level
    agreement over all cells from the stored outputs, plus B-level
    agreement on the recomputed seed."""
    from scipy.stats import spearmanr
    from t51_cluster_stats import fam_faith_expanded
    cells2 = pd.read_csv(OUT / "probe2_cells.csv")
    qual: dict = {}
    for ds in DATASETS:
        C = pd.read_csv(FAITH / f"C_pooled_corr_{ds}.csv", index_col=0)
        p8 = partition(C, 0.8)
        feats = list(C.index)
        P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                        ).reindex(index=feats, columns=feats)
        D = pd.read_csv(FAITH / f"D_retrained_{ds}_seed1_cpu_t2.csv",
                        index_col=0)
        m5 = measure_host(ds, 1, "SNI", {0.8: p8}, n_rep=5)
        m10 = measure_host(ds, 1, "SNI", {0.8: p8}, n_rep=10)
        rows = []
        b_agree = []
        A1 = pd.read_csv(FAITH / f"A_{ds}_seed1_cpu_t2.csv", index_col=0)
        for f in m5["cells"][0.8]:
            gidx = sorted(m5["cells"][0.8][f])
            grp_members = [[c for c in p8[i] if c != f] for i in gidx]
            B5 = [m5["cells"][0.8][f][i] for i in gidx]
            B10 = [m10["cells"][0.8][f][i] for i in gidx]
            gd = group_scores(D.loc[f], grp_members)
            gp = group_scores(P.loc[f], grp_members)
            rows.append({
                "target": f,
                "rho_D_5rep": round(rho_cells(B5, gd), 4),
                "rho_D_10rep": round(rho_cells(B10, gd), 4),
                "rho_TAP_5rep": round(rho_cells(B5, gp), 4),
                "rho_TAP_10rep": round(rho_cells(B10, gp), 4)})
            aref = [float(A1.loc[f, g[0]]) for g in grp_members]
            b_agree.append(round(float(spearmanr(B5, aref).statistic), 4))
        d5 = np.array([r["rho_D_5rep"] - r["rho_TAP_5rep"] for r in rows])
        d10 = np.array([r["rho_D_10rep"] - r["rho_TAP_10rep"] for r in rows])
        # rho-level agreement over every stored cell (all 15 seeds)
        sub = cells2[(cells2.dataset == ds) & (cells2.variant == "SNI")
                     & (cells2.tau == 0.8)]
        fam1 = fam_faith_expanded(ds, "SNI")
        pairs = []
        for s, arr in fam1.items():
            targets = sorted(pd.read_csv(
                FAITH / f"A_{ds}_seed{s}_cpu_t2.csv", index_col=0).index)
            p1 = dict(zip(targets, arr))          # Delta rho, probe 1
            for _, r in sub[sub.seed == s].iterrows():
                pairs.append((p1[r.target],
                              float(r.rho_group_D - r.rho_group_TAP)))
        pr = np.array(pairs)
        qual[ds] = {
            "rep_sensitivity_seed1": rows,
            "delta_median_5rep": round(float(np.median(d5)), 4),
            "delta_median_10rep": round(float(np.median(d10)), 4),
            "B_level_agreement_seed1_spearman_per_target": b_agree,
            "rho_level_agreement_all_cells": {
                "n_cells": int(len(pr)),
                "spearman_delta_probe1_vs_probe2":
                    round(float(spearmanr(pr[:, 0], pr[:, 1]).statistic), 4),
                "median_abs_diff":
                    round(float(np.median(np.abs(pr[:, 0] - pr[:, 1]))), 4)}}
    (OUT / "probe2_qualifiers.json").write_text(json.dumps(qual, indent=1))
    for ds, q in qual.items():
        print(f"{ds}: seed-1 Delta median 5rep {q['delta_median_5rep']:+.4f} "
              f"-> 10rep {q['delta_median_10rep']:+.4f}; B-agreement "
              f"median {np.median(q['B_level_agreement_seed1_spearman_per_target']):.3f}; "
              f"rho-agreement r={q['rho_level_agreement_all_cells']['spearman_delta_probe1_vs_probe2']} "
              f"(median |diff| {q['rho_level_agreement_all_cells']['median_abs_diff']})")
    print(f"[ok] wrote {OUT / 'probe2_qualifiers.json'}")
    return 0


# ---------------- selftest -------------------------------------------------- #
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    feats = ["a", "b", "c", "d"]
    C = pd.DataFrame(0.0, index=feats, columns=feats)
    C.loc["a", "b"] = C.loc["b", "a"] = 0.9
    p8 = partition(C, 0.8)
    check(p8 == [["a", "b"], ["c"], ["d"]],
          "partition at 0.8: one pair + two singletons, first-member order")
    check(partition(C, 0.95) == [["a"], ["b"], ["c"], ["d"]],
          "partition above the edge: all singletons")
    C.loc["b", "c"] = C.loc["c", "b"] = 0.85
    check(partition(C, 0.8) == [["a", "b", "c"], ["d"]],
          "chained edges merge transitively")
    row = pd.Series({"a": 1.0, "b": 3.0, "c": 5.0, "d": 0.0})
    check(group_scores(row, [["a", "b"], ["c"]]) == [2.0, 5.0]
          and group_scores(row, [["a", "b"]], how="sum") == [4.0],
          "group scores: member mean primary, sum sensitivity")
    check(abs(rho_cells([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-12
          and abs(rho_cells([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12,
          "rho over cells: monotone 1 / -1")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                choices=["run", "qualifiers", "selftest"])
    ap.add_argument("--hosts", nargs="*", default=["SNI", "NoPrior"])
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "qualifiers":
        return stage_qualifiers()
    return stage_run(a.hosts)


if __name__ == "__main__":
    raise SystemExit(main())
