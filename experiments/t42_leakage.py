"""T4.2 -- leakage detection, the fifth VERA axis.

Implements docs/T42_leakage_rules.md (committed 6535787, addendum a69eea8)
verbatim: five proxy classes (exact; noisy rho ~0.95/0.80/0.60; interaction;
consequence; discrepancy control) x 3 host seeds x 2 datasets, a permutation null per
condition, and >= 20 random-proxy calibration runs per dataset. Detection at
FPR alpha = 0.05 against the empirical 95% quantile of the calibration
distribution for the same object, position (target row) and dataset. Counts
per (object x condition) out of 6 with Wilson 95% intervals; classes never
pooled; interaction stop threshold D >= 3/6 while TAP <= 1/6; the discrepancy
control read under both estimands.

Stages:
    plan      -- enumerate runs, print scale / machine-hours (launch-day gate)
    run       -- execute one worker slice (--slice i/n), resume-by-artifact
    analyze   -- calibration quantiles, detections, Wilson CIs, stop flags
    selftest  -- known-answer fixtures for constructions and analyzer

    env PYTHONHASHSEED=2025 python experiments/t42_leakage.py --stage selftest
    env PYTHONHASHSEED=2025 python experiments/t42_leakage.py --stage plan
    env PYTHONHASHSEED=2025 python experiments/t42_leakage.py --stage run --slice 0/10
    env PYTHONHASHSEED=2025 python experiments/t42_leakage.py --stage analyze
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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

FAITH = CODE_ROOT / "results" / "T3_faithfulness"
OUT = CODE_ROOT / "results" / "T4_leakage"

DATASETS = ["MIMIC", "eICU"]
HOST_SEEDS = [1, 2, 3]
# T4.2 confirmatory replication (docs/T42_confirmatory_replication_rules.md,
# commit 29f3a81): interaction only, frozen gen-3 construction, fresh hosts.
CONF_SEEDS = [5, 8, 13]


def _seed_index(s: int) -> int:
    """Global host-seed index: originals 0-2, confirmatory 3-5. Keeps every
    original RNG stream byte-identical while giving each confirmatory seed a
    fresh, prospectively specified stream (rules commit 29f3a81)."""
    return (HOST_SEEDS + CONF_SEEDS).index(s)
# The seven injected conditions. "noisy" appears at three strengths; the five
# classes of the rule are the five distinct kinds. Reported per condition,
# never pooled (rule: "classes NEVER pooled into one number").
CONDITIONS = [("exact", None), ("noisy", 0.95), ("noisy", 0.80),
              ("noisy", 0.60), ("interaction", None), ("consequence", None),
              ("decoy", None)]
N_CALIB_PER_SEED = 7            # 7 x 3 host seeds = 21 >= 20 per dataset
ALPHA = 0.05                    # fixed in the rule
PROXY = "proxy_injected"
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest", "Permutation-on-SNI"]
N_PERM = 5                      # Perm-on-SNI readout repeats (T3.2 convention)
# The 20% re-report gate compares against the reference fixed in the rule
# addendum's erratum (docs/T42_leakage_rules.md, commit fcc8f85): 124
# retrains at the P4_T41 planning figures. These are committed-rule
# constants (the prospectively specified reference point), not a measurement.
RULE_REF = {"retrains": 124, "sec": {"MIMIC": 5500.0, "eICU": 3720.0}}


def baseline_sec() -> tuple:
    """Canary baseline, runtime-read from archived timing records
    (P4K-A ruling 1.3: no hand-copied constants). Source: the five T3.2
    host-retrain walls per dataset (meta_*.json, wall_train_sec), the only
    stored same-shape retrains. They were measured beside the running grid,
    so the 1.5x canary is lenient (it can under-trigger, never falsely
    kill T4.2); direction documented in the receipt."""
    walls, paths = {}, {}
    for ds in DATASETS:
        vs, ps = [], []
        for s in [1, 2, 3, 5, 8]:
            p = FAITH / f"meta_{ds}_seed{s}_cpu_t{_NT}.json"
            rec = json.loads(p.read_text())
            v = float(rec["wall_train_sec"])
            assert v > 0, f"non-positive wall in {p}"
            vs.append(v)
            ps.append(str(p))
        assert len(vs) == 5, f"{ds}: expected 5 archived walls, got {len(vs)}"
        walls[ds] = float(np.median(vs))
        paths[ds] = ps
    return walls, paths


def _cond_name(cls: str, rho) -> str:
    return f"{cls}{int(rho * 100)}" if cls == "noisy" else cls


# --------------------------------------------------------------------------- #
# target selection -- deterministic from stored T3.2 artifacts
# --------------------------------------------------------------------------- #
def attack_targets(ds: str, cont: list) -> list:
    """Top-3 continuous targets by median stored ablation effect (T3.2 A
    matrices, seeds 1,2,3,5,8): host seed k attacks rank k. Deterministic."""
    eff = {}
    for f in cont:
        vals = []
        for s in [1, 2, 3, 5, 8]:
            A = pd.read_csv(FAITH / f"A_{ds}_seed{s}_cpu_t{_NT}.csv",
                            index_col=0)
            if f in A.index:
                vals.append(float(np.nanmedian(A.loc[f].to_numpy(float))))
        if vals:
            eff[f] = float(np.median(vals))
    ranked = sorted(eff, key=lambda f: -eff[f])
    if len(ranked) < 3:
        raise RuntimeError(f"{ds}: fewer than 3 continuous targets with "
                           f"stored ablation rows ({ranked})")
    return ranked[:3]


# --------------------------------------------------------------------------- #
# proxy constructions -- each asserts its own class definition
# --------------------------------------------------------------------------- #
def _z(v: np.ndarray) -> np.ndarray:
    sd = float(v.std()) or 1.0
    return (v - float(v.mean())) / sd


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _partners_ranked(complete: pd.DataFrame, cont: list, t: str,
                     mode: str) -> list:
    """Continuous partners ranked for the class: ascending |corr| for
    interaction (want ~independent), descending for the discrepancy
    control (want strong).
    A ranked list, not a single pick: real columns can defeat a
    construction's assertions (skew, ties), and the fallback is the next
    candidate, deterministically, with the choice recorded in run meta."""
    tv = pd.to_numeric(complete[t], errors="coerce").to_numpy(float)
    cs = {}
    for q in cont:
        if q == t:
            continue
        qv = pd.to_numeric(complete[q], errors="coerce").to_numpy(float)
        cs[q] = abs(_corr(tv, qv))
    return sorted(cs, key=cs.get, reverse=(mode == "max"))


def make_proxy(cls: str, rho, t_vals: np.ndarray, q_vals, rng,
               info: dict | None = None) -> np.ndarray:
    """Build one proxy column; assert the class's defining property.
    `info`, when given, receives construction provenance (P4N §2.2):
    generation tag, residualization beta, split balance."""
    zt = _z(t_vals)
    n = len(zt)
    if cls == "exact":
        p = t_vals.copy().astype(float)
        assert _corr(p, t_vals) > 0.9999, "exact duplicate must correlate 1"
        return p
    if cls == "noisy":
        p = rho * zt + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n)
        c = _corr(p, t_vals)
        assert abs(c - rho) < 0.05, f"noisy proxy corr {c:.3f} != rho {rho}"
        return p
    if cls == "interaction":
        zq = _z(q_vals)
        # Median split, not sign-at-zero: real clinical columns are skewed,
        # so sign(z_q) is imbalanced and E[z_t^2 sign(z_q)] != 0 even with
        # q independent of t -- exactly how six workers died on 2026-08-25
        # (the assertion below correctly refused the mislabeled proxy; the
        # construction, not the guard, was wrong). A median split balances
        # the +-1 mass regardless of shape, and with q independent of t the
        # expectation factorises, so the parity argument needs no symmetry.
        sgn = np.where(zq > float(np.median(zq)), 1.0, -1.0)
        bal = abs(float(sgn.mean()))
        assert bal < 0.20, f"partner sign split imbalanced ({bal:.2f}; ties)"
        p = zt * sgn + 0.15 * rng.standard_normal(n)
        # Linear residualization on t (2026-08-26, MIMIC_s3): the median
        # split balances the sign mass, but clinical columns also couple at
        # the variance level (heteroskedasticity), so E[z_t^2 sgn] can
        # exceed the guard for EVERY partner. Removing the linear-in-t
        # component makes marginal independence hold by construction;
        # joint predictivity survives (p*sgn = z_t(1 - beta*sgn) + noise,
        # |beta| < 1). The guards below verify rather than trust this.
        beta = float(np.polyfit(zt, p, 1)[0])
        p = p - beta * zt
        if info is not None:
            info.update({"construction_generation": "gen3-residualized",
                         "beta_residualisation": round(beta, 6),
                         "sign_split_balance": round(bal, 4)})
        c = _corr(p, t_vals)
        assert abs(c) < 0.10, f"interaction proxy marginally corr {c:.3f}"
        joint = _corr(p * sgn, t_vals)
        assert joint > 0.5, f"interaction proxy not jointly predictive {joint:.3f}"
        # Non-degeneracy: with a t-correlated partner, z_t ~ z_q makes
        # p ~ |z_q| -- a function of the partner alone (a disguised
        # discrepancy control;
        # marginal independence survives by parity and cannot catch this).
        dg = _corr(p, np.abs(zq))
        assert abs(dg) < 0.20, f"interaction proxy degenerates to |partner| ({dg:.3f})"
        return p
    if cls == "consequence":
        p = (zt > 0).astype(float) + 0.10 * rng.standard_normal(n)
        c = _corr(p, t_vals)
        assert 0.30 < abs(c) < 0.98, f"consequence field corr {c:.3f} out of band"
        return p
    if cls == "decoy":
        zq = _z(q_vals)
        p = zq + 0.30 * rng.standard_normal(n)
        c = _corr(p, t_vals)
        assert abs(c) > 0.10, f"decoy not marginally correlated ({c:.3f})"
        # no incremental information beyond q: partial corr(p, t | q) ~ 0
        rp = p - np.polyval(np.polyfit(zq, p, 1), zq)
        rt = t_vals - np.polyval(np.polyfit(zq, t_vals, 1), zq)
        pc = _corr(rp, rt)
        assert abs(pc) < 0.10, f"decoy carries incremental info (partial {pc:.3f})"
        return p
    raise ValueError(cls)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan() -> list:
    """Full run list: [(tag, ds, host_seed, kind, cond, rho)]; kind in
    {inj, null, calib}."""
    runs = []
    for ds in DATASETS:
        for s in HOST_SEEDS:
            for cls, rho in CONDITIONS:
                cn = _cond_name(cls, rho)
                runs.append((f"{ds}_s{s}_{cn}_inj", ds, s, "inj", cls, rho))
                runs.append((f"{ds}_s{s}_{cn}_null", ds, s, "null", cls, rho))
            for k in range(N_CALIB_PER_SEED):
                runs.append((f"{ds}_s{s}_calib{k}", ds, s, "calib", None, None))
    return runs


def conf_plan() -> list:
    """Confirmatory replication run list (rules commit 29f3a81): interaction
    only, host seeds 5/8/13, inj + null; 12 host retrains, no new
    calibration (thresholds reused from the original pool)."""
    runs = []
    for ds in DATASETS:
        for s in CONF_SEEDS:
            runs.append((f"{ds}_s{s}_interaction_inj", ds, s, "inj",
                         "interaction", None))
            runs.append((f"{ds}_s{s}_interaction_null", ds, s, "null",
                         "interaction", None))
    return runs


def stage_plan() -> int:
    runs = plan()
    per_ds = {ds: sum(1 for r in runs if r[1] == ds) for ds in DATASETS}
    # Prospectively specified 20% gate: BOTH sides in the rule's caliber (addendum
    # erratum): counts vs 124, machine-hours vs 124 x rule-fixed per-run sec.
    hours_rule = sum(per_ds[ds] * RULE_REF["sec"][ds]
                     for ds in DATASETS) / 3600.0
    ref_hours = RULE_REF["retrains"] * float(
        np.mean(list(RULE_REF["sec"].values()))) / 3600.0
    walls, paths = baseline_sec()
    hours_arch = sum(per_ds[ds] * walls[ds] for ds in DATASETS) / 3600.0
    done = sum(1 for r in runs if (OUT / "runs" / r[0] / "scores.json").exists())
    print(f"T4.2 plan: {len(runs)} host retrains "
          f"({per_ds}, = 42 inj + 42 null + 42 calib), {done} already done")
    print(f"rule-caliber estimate {hours_rule:.1f} machine-h "
          f"(reference {ref_hours:.1f}); wall @10 workers ~ {hours_rule/10:.1f} h")
    print(f"archived-walls projection (grid-contended era, upper band): "
          f"{hours_arch:.1f} machine-h; canary baselines "
          f"{ {d: round(w) for d, w in walls.items()} } from "
          f"{len(paths['MIMIC']) + len(paths['eICU'])} meta files")
    print(f"20% re-report trigger (addendum a69eea8 + erratum): "
          f"{'WITHIN' if len(runs) <= RULE_REF['retrains'] * 1.2 and hours_rule <= ref_hours * 1.2 else 'ABOVE - REPORT FIRST'}")
    return 0


# --------------------------------------------------------------------------- #
# one run: inject -> retrain host -> six object matrices -> scores
# --------------------------------------------------------------------------- #
def run_case(tag: str, ds: str, host_seed: int, kind: str, cls, rho,
             out_root: "Path | None" = None, both_signals: bool = False):
    """Returns the host-retrain wall seconds (train only), or None if the
    run was already on disk. Train-only is the sentinel's caliber on BOTH
    sides (the archived baseline is T3.2's wall_train_sec).

    `both_signals` (T6.1 addendum 2026-08-29d SS4) additionally emits the
    Permutation-on-SNI readout with its error signal taken from the host's own
    completed table instead of the withheld values. Both come from the SAME
    freshly trained host, so the pair is within-host and the host draw cancels
    -- comparing a fresh no-oracle matrix against the ARCHIVED oracle one
    would confound the oracle with the training draw, which is the very defect
    this addendum exists to remove. It costs almost nothing: the expensive
    part is the permuted forward passes, and both variants score the same
    predictions against different targets.

    `out_root` redirects every write, so a recompute cannot touch the archived
    campaign.
    """
    import torch
    import yaml
    from sklearn.preprocessing import StandardScaler
    from common import determinism
    from prior_attribution import compute_P, load_real_case
    from pilot_r21 import run_missforest_family
    from sni.imputer import SNIConfig, SNIImputer

    rdir = (out_root or OUT) / "runs" / tag
    if (rdir / "scores.json").exists():
        print(f"[cached] {tag}", flush=True)
        return None
    rdir.mkdir(parents=True, exist_ok=True)

    missing, mask_df, cat, cont = load_real_case(ds)
    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")[list(missing.columns)]
    targets3 = attack_targets(ds, cont)
    # mod-3 rotation extends the original mapping unchanged (indices 0-2)
    # to the confirmatory seeds (3-5 -> the same three ranked targets).
    t = targets3[_seed_index(host_seed) % len(targets3)]
    tv = pd.to_numeric(complete[t], errors="coerce").to_numpy(float)

    meta = {"tag": tag, "dataset": ds, "host_seed": host_seed, "kind": kind,
            "condition": _cond_name(cls, rho) if cls else "calib",
            "target": t, "targets_ranked": targets3}
    rng = np.random.default_rng(
        50_000 + 1000 * _seed_index(host_seed)
        + 10 * DATASETS.index(ds) + hash(meta["condition"]) % 997)
    if kind == "calib":
        k = int(tag.rsplit("calib", 1)[1])
        prng = np.random.default_rng(9000 + 100 * host_seed + k)
        proxy = prng.standard_normal(len(complete))
        meta["partner"] = None
    else:
        if cls in ("interaction", "decoy"):
            ranked = _partners_ranked(complete, cont, t,
                                      "min" if cls == "interaction" else "max")
            proxy, last, rejected = None, None, []
            cinfo: dict = {}
            for q in ranked[:5]:
                qv = pd.to_numeric(complete[q], errors="coerce"
                                   ).to_numpy(float)
                try:
                    proxy = make_proxy(cls, rho, tv, qv, rng, info=cinfo)
                    meta["partner"] = q
                    break
                except AssertionError as exc:
                    last = exc
                    rejected.append({"partner": q, "reason": str(exc)})
                    print(f"[partner-fallback] {tag}: {q} rejected ({exc})",
                          flush=True)
            if proxy is None:
                raise AssertionError(
                    f"{tag}: no admissible partner among first 5 ({last})")
            meta["partner_fallback_rejected"] = rejected
            meta.update(cinfo)               # P4N SS2.2: beta + generation
        else:
            meta["partner"] = None
            proxy = make_proxy(cls, rho, tv, None, rng)
        if kind == "null":
            nrng = np.random.default_rng(777 * host_seed
                                         + hash(meta["condition"]) % 997)
            proxy = proxy[nrng.permutation(len(proxy))]
        meta["proxy_corr_with_target"] = _corr(proxy, tv)

    cont_i = cont + [PROXY]
    feats_i = cat + cont_i
    complete_i = complete.copy()
    complete_i[PROXY] = proxy
    missing_i = missing.copy()
    missing_i[PROXY] = proxy                      # fully observed
    mask_i = mask_df.copy()
    mask_i[PROXY] = False
    missing_i, mask_i = missing_i[feats_i], mask_i[feats_i]

    mats: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    mats["P"] = compute_P(cat, cont_i, host_seed, missing_i, mask_i)
    fam, _no_model = run_missforest_family(missing_i, cat, cont_i,
                                           host_seed, feats_i)
    for name, (M, _s1, _s2) in fam.items():
        mats[name] = M

    proto = yaml.safe_load((CODE_ROOT / "configs" / "training_protocol.yaml"
                            ).read_text())["protocol"]
    determinism.apply("deterministic", seed=host_seed)
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont_i,
                     config=SNIConfig(seed=host_seed, use_gpu=False))
    imp.cfg.epochs = int(proto["epochs"]["SNI"])
    imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
    tt = time.time()
    X_final = imp.impute(X_missing=missing_i, X_complete=None, mask_df=mask_i)
    wall_train = time.time() - tt
    mats["SNI-D"] = imp.compute_dependency_matrix()

    # Permutation-on-SNI readout: T3.2's ablation algorithm on the injected
    # table (truth from the complete column; proxy is never a target).
    tgt_rows = [f for f in feats_i if int(mask_i[f].sum()) > 0]
    A = pd.DataFrame(np.nan, index=tgt_rows, columns=feats_i)
    A_no = pd.DataFrame(np.nan, index=tgt_rows, columns=feats_i)
    n = len(X_final)
    for f in tgt_rows:
        fi = feats_i.index(f)
        Z_df = X_final.drop(columns=[f])
        srcs = list(Z_df.columns)
        Z_enc, _ = imp._encode_dataframe_for_training(Z_df)
        Z_enc = np.nan_to_num(Z_enc, nan=0.0)
        present = ~mask_i[f].to_numpy(dtype=bool)
        miss_idx = np.where(~present)[0]
        scaler_Z = StandardScaler().fit(Z_enc[present])
        Z_s = scaler_Z.transform(Z_enc)
        y_obs = pd.to_numeric(X_final[f], errors="coerce"
                              ).to_numpy(float)[present]
        scaler_y = StandardScaler().fit(y_obs.reshape(-1, 1))
        truth = pd.to_numeric(complete_i[f], errors="coerce"
                              ).to_numpy(float)[miss_idx]
        sd = float(truth.std()) or 1.0
        # the same rows read from the host's own completion instead of the
        # withheld values: the no-oracle error target (T6.1 SS4)
        truth_no = pd.to_numeric(X_final[f], errors="coerce"
                                 ).to_numpy(float)[miss_idx]
        sd_no = float(truth_no.std()) or 1.0
        model = imp.models[f]
        model.eval()

        def _predict(Zmat: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                yh = model(torch.tensor(Zmat[miss_idx],
                                        dtype=torch.float32))[0]
            return scaler_y.inverse_transform(
                yh.cpu().numpy().reshape(-1, 1)).flatten()

        yh0 = _predict(Z_s)
        e0 = float(np.sqrt(np.mean((truth - yh0) ** 2))) / sd
        e0_no = float(np.sqrt(np.mean((truth_no - yh0) ** 2))) / sd_no
        for jj, j in enumerate(srcs):
            deltas, deltas_no = [], []
            for r in range(N_PERM):
                pr = np.random.default_rng(
                    20_000 * host_seed + 100 * fi + 5 * jj + r)
                Zp = Z_s.copy()
                Zp[:, jj] = Zp[pr.permutation(n), jj]
                yhp = _predict(Zp)
                e = float(np.sqrt(np.mean((truth - yhp) ** 2))) / sd
                deltas.append(e - e0)
                if both_signals:
                    e_no = float(np.sqrt(np.mean((truth_no - yhp) ** 2))) / sd_no
                    deltas_no.append(e_no - e0_no)
            A.loc[f, j] = float(np.mean(deltas))
            if both_signals:
                A_no.loc[f, j] = float(np.mean(deltas_no))
    mats["Permutation-on-SNI"] = A
    if both_signals:
        mats["Permutation-on-SNI-noOracle"] = A_no

    target_scores, row_ranks = {}, {}
    for obj, M in mats.items():
        M.to_csv(rdir / f"M_{obj}.csv")
        ts_, rr_ = {}, {}
        for f in M.index:
            if PROXY not in M.columns:
                raise RuntimeError(f"{obj}: no {PROXY} column")
            row = M.loc[f]
            if pd.isna(row.get(PROXY)):
                continue
            vals = row.drop(labels=[f], errors="ignore").astype(float)
            ts_[f] = float(row[PROXY])
            rr_[f] = int((vals > float(row[PROXY])).sum() + 1)  # 1 = top
        target_scores[obj], row_ranks[obj] = ts_, rr_

    meta.update({"wall_train_sec": round(wall_train, 1),
                 "wall_total_sec": round(time.time() - t0, 1),
                 "n_target_rows": len(tgt_rows)})
    (rdir / "scores.json").write_text(json.dumps(
        {"meta": meta, "target_scores": target_scores,
         "row_ranks": row_ranks}, indent=1))
    print(f"[ok] {tag}  train={wall_train:.0f}s total={time.time()-t0:.0f}s",
          flush=True)
    return wall_train


def _alive_grid_shards() -> int:
    r = subprocess.run(["pgrep", "-fc", r"run_grid\.py --queue cpu"],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def _self_baseline(ds: str, runs_dir: Path):
    """P4K-B self-calibrated baseline: median of T4.2's OWN first three
    completed retrain walls for this dataset, completion order = scores.json
    mtime, all workers pooled. None until three samples exist (until then
    only the archived condition applies, P4K-B item 3)."""
    recs = []
    for p in runs_dir.glob("*/scores.json"):
        m = json.loads(p.read_text())["meta"]
        if m.get("dataset") == ds and m.get("wall_train_sec") is not None:
            recs.append((p.stat().st_mtime, m["tag"],
                         float(m["wall_train_sec"])))
    recs.sort()
    if len(recs) < 3:
        return None
    first3 = recs[:3]
    walls = [w for _t, _g, w in first3]
    assert len(walls) == 3 and all(w > 0 for w in walls), \
        f"self-baseline {ds}: bad walls {walls}"
    return {"median_sec": float(np.median(walls)),
            "source_runs": [g for _t, g, _w in first3],
            "walls": walls}


def _is_slow(train_wall: float, archived: float, self_med) -> list:
    """P4K-B OR trigger: a run is slow if it exceeds 1.5x the archived
    baseline OR 1.5x the self-calibrated one (once that exists). Returns
    the breached condition names (empty = not slow). Both are triplines:
    stop + investigate + report, never a verdict gate."""
    hit = []
    if train_wall > 1.5 * archived:
        hit.append(f"archived({archived:.0f}s)")
    if self_med is not None and train_wall > 1.5 * float(self_med):
        hit.append(f"self({float(self_med):.0f}s)")
    return hit


def stage_run(sl: str, planfn=plan) -> int:
    i, nw = (int(x) for x in sl.split("/"))
    alive = _alive_grid_shards()
    if alive + nw > 12:
        print(f"REFUSING TO RUN: {alive} grid shards alive + {nw} T4.2 "
              f"workers > 12 (addendum a69eea8 envelope).", file=sys.stderr)
        return 2
    walls, wall_paths = baseline_sec()      # P4K-A 1.3: archived, asserted
    wfile = OUT / f"window_{i}.json"
    selfb: dict = {}                        # P4K-B: per-dataset, refreshed

    def _window(status: str) -> None:
        rec = (json.loads(wfile.read_text()) if wfile.exists()
               else {"worker": sl, "start_unix": time.time(),
                     "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "canary_baseline_sec": walls,
                     "canary_baseline_sources": wall_paths})
        rec.update({"end_unix": time.time(),
                    "end": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                    "self_baseline": selfb})
        wfile.write_text(json.dumps(rec, indent=1))

    _window("running")
    breaches = 0
    last_hits: list = []
    failed: list = []
    rc = 0
    for (tag, ds, s, kind, cls, rho) in planfn()[i::nw]:
        if list(OUT.glob("PAUSE_*.flag")):
            print("PAUSE flag present; worker stops.", file=sys.stderr)
            rc = 3
            break
        # Per-run containment (added 2026-08-25 after six workers died on
        # one construction assertion and took their whole slices with
        # them): a failed run is logged and skipped -- run_grid's pattern.
        # analyze still refuses while any run is missing, so nothing fails
        # silently; failures surface there and in the [FAIL] lines.
        try:
            tw = run_case(tag, ds, s, kind, cls, rho)
        except Exception as exc:
            import traceback as _tb
            print(f"[FAIL {tag}] {exc!r}\n{_tb.format_exc()[-600:]}",
                  file=sys.stderr, flush=True)
            failed.append(tag)
            continue
        if tw is not None:                           # a real (uncached) run
            sb = _self_baseline(ds, OUT / "runs")
            selfb[ds] = sb
            hits = _is_slow(tw, walls[ds], sb["median_sec"] if sb else None)
            breaches = breaches + 1 if hits else 0
            last_hits = hits or last_hits
            if breaches >= 2:
                # Tripline, not a gate (P4K-A 1.2): stop + investigate +
                # report; nothing about any completed result is vetoed.
                (OUT / f"PAUSE_{i}.flag").write_text(
                    f"{tag}: train {tw:.0f}s breached {last_hits} twice in "
                    f"a row (archived sources: {wall_paths[ds][0]} ...; "
                    f"self-baseline: {sb}). Tripline: stop + investigate + "
                    f"report -- not a verdict gate.")
                print(f"CANARY: two consecutive retrains slow "
                      f"({last_hits}); pausing all workers "
                      f"(tripline, not a verdict gate).", file=sys.stderr)
                rc = 3
                break
        _window("running")
    if failed:
        print(f"worker {sl}: {len(failed)} runs FAILED: {failed}",
              file=sys.stderr, flush=True)
        rc = rc or 1
    _window("paused" if rc == 3 else ("done-with-failures" if failed
                                      else "done"))
    return rc


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def _load_scores(rdir: Path) -> dict:
    return json.loads((rdir / "scores.json").read_text())


def _calib_q95(base: Path, runs: list) -> dict:
    """Calibration quantiles per (object, dataset, target row), pooled over
    the run list's calib entries. Raises ValueError below the prospectively specified
    n >= 20 gate. (Factored out of stage_analyze unchanged so the
    confirmatory analyzer reuses the ORIGINAL pool byte-identically.)"""
    q95: dict = {}
    for ds in DATASETS:
        pool: dict = {}
        for (tag, d2, s, kind, *_x) in runs:
            if d2 != ds or kind != "calib":
                continue
            sc = _load_scores(base / "runs" / tag)["target_scores"]
            for obj in OBJECTS:
                for f, v in sc.get(obj, {}).items():
                    pool.setdefault((obj, f), []).append(float(v))
        for (obj, f), vals in pool.items():
            if len(vals) < 20:
                raise ValueError(f"REFUSING TO ANALYZE: calibration "
                                 f"n={len(vals)} < 20 for ({obj}, {ds}, {f})")
            q95[(obj, ds, f)] = float(np.quantile(vals, 1.0 - ALPHA))
    return q95


def mcnemar_exact_pairs(pairs: list) -> dict:
    """Exact two-sided McNemar on (detected_D, detected_TAP) pairs: binomial
    on the discordant count, p = min(1, 2 * P(X <= min(b, c) | n, 1/2))."""
    from math import comb
    b = sum(1 for x, y in pairs if x and not y)          # D-only
    c = sum(1 for x, y in pairs if y and not x)          # TAP-only
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(
        comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)
    return {"n_pairs": len(pairs), "D_only": b, "TAP_only": c,
            "p_two_sided": round(p, 6)}


def binom_tail(k: int, n: int, p: float) -> float:
    """Exact upper tail P(X >= k | n, p)."""
    from math import comb
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def stage_analyze(base: Path = OUT) -> int:
    runs = plan()
    missing_runs = [t for (t, *_r) in runs
                    if not (base / "runs" / t / "scores.json").exists()]
    if missing_runs:
        print(f"REFUSING TO ANALYZE: {len(missing_runs)} of {len(runs)} runs "
              f"missing (first: {missing_runs[:3]})", file=sys.stderr)
        return 2

    # calibration quantiles per (object, dataset, target row)
    try:
        q95 = _calib_q95(base, runs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    det_rows, curve_rows = [], []
    counts: dict = {}
    for (tag, ds, s, kind, cls, rho) in runs:
        if kind == "calib":
            continue
        rec = _load_scores(base / "runs" / tag)
        t = rec["meta"]["target"]
        cn = rec["meta"]["condition"]
        for obj in OBJECTS:
            score = rec["target_scores"].get(obj, {}).get(t)
            if score is None:
                continue
            thr = q95[(obj, ds, t)]
            det = bool(score > thr)
            gen = (("gen3" if "beta_residualisation" in rec["meta"]
                    else "gen2") if cn == "interaction" else None)
            det_rows.append({"object": obj, "dataset": ds, "host_seed": s,
                             "condition": cn, "kind": kind, "target": t,
                             "score": score, "q95": thr, "detected": det,
                             "generation": gen})
            counts.setdefault((obj, cn, kind), []).append(det)
            curve_rows.append({"object": obj, "dataset": ds, "host_seed": s,
                               "condition": cn, "kind": kind,
                               "rank_in_row": rec["row_ranks"][obj].get(t)})

    summary_counts = []
    for (obj, cn, kind), dets in sorted(counts.items()):
        k, n = sum(dets), len(dets)
        lo, hi = wilson(k, n)
        summary_counts.append({"object": obj, "condition": cn, "kind": kind,
                               "detected": k, "n": n,
                               "wilson_lo": round(lo, 4),
                               "wilson_hi": round(hi, 4)})

    def _k(obj, cn, kind="inj"):
        e = [r for r in summary_counts
             if r["object"] == obj and r["condition"] == cn
             and r["kind"] == kind]
        return e[0]["detected"] if e else 0

    stop_interaction = bool(_k("SNI-D", "interaction") >= 3
                            and _k("P", "interaction") <= 1)
    # P4N SS2.1: interaction results decomposed by construction generation.
    # Descriptive only -- no verdict gate keys on it; the class verdict uses
    # the pooled counts above, and any substantive divergence between
    # generations is a reporting qualifier.
    by_gen: dict = {}
    for r in det_rows:
        if r["condition"] != "interaction" or r["kind"] != "inj":
            continue
        g = by_gen.setdefault(r["generation"], {"n_by_dataset": {},
                                                "detected_by_object": {}})
        if r["object"] == OBJECTS[0]:
            g["n_by_dataset"][r["dataset"]] = \
                g["n_by_dataset"].get(r["dataset"], 0) + 1
        g["detected_by_object"][r["object"]] = \
            g["detected_by_object"].get(r["object"], 0) + int(r["detected"])
    decoy_fp = {obj: _k(obj, "decoy") for obj in OBJECTS}
    null_rate = {obj: round(float(np.mean(
        [r["detected"] for r in det_rows
         if r["object"] == obj and r["kind"] == "null"])), 4)
        for obj in OBJECTS}
    # P5R SS4.2: the observed null rates get a formal check -- exact
    # binomial upper tail P(X >= k | n, alpha) against the nominal
    # alpha = 0.05 of the threshold-calibrated detection design. Descriptive (additive field;
    # every previously shipped field is byte-unchanged).
    null_binom = {}
    for obj in OBJECTS:
        dets = [r["detected"] for r in det_rows
                if r["object"] == obj and r["kind"] == "null"]
        null_binom[obj] = {"detected": int(sum(dets)), "n": len(dets),
                           "p_geq_k_exact_binomial":
                               round(binom_tail(int(sum(dets)),
                                                len(dets), ALPHA), 6)}
    # Chat ruling 2026-08-29 (P5R SS4 writing obligation): the same-host
    # probe's interaction detections must always be stated together with
    # its above-nominal FPR, plus this bound -- computed here, never
    # hand-copied: even at the IMPLEMENTED false-positive rate, the
    # probability of ALL interaction injections being detected by chance
    # alone is fpr^n.
    pr_null = [r["detected"] for r in det_rows
               if r["object"] == "Permutation-on-SNI"
               and r["kind"] == "null"]
    pr_int = [r["detected"] for r in det_rows
              if r["object"] == "Permutation-on-SNI"
              and r["condition"] == "interaction" and r["kind"] == "inj"]
    fpr = float(np.mean(pr_null))
    probe_bound = {"object": "Permutation-on-SNI",
                   "fpr_implemented": round(fpr, 6),
                   "fpr_counts": f"{int(sum(pr_null))}/{len(pr_null)}",
                   "interaction_detected": int(sum(pr_int)),
                   "interaction_n": len(pr_int),
                   "p_all_detected_at_implemented_fpr":
                       float(fpr ** len(pr_int))}

    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(det_rows).to_csv(base / "t42_detection.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(base / "t42_rank_curves.csv", index=False)
    (base / "t42_summary.json").write_text(json.dumps(
        {"alpha": ALPHA, "counts": summary_counts,
         "stop_report_interaction_D_wins": stop_interaction,
         "interaction_counts": {o: _k(o, "interaction") for o in OBJECTS},
         "interaction_by_generation": by_gen,
         "decoy_false_positives": decoy_fp,
         "null_detection_rate": null_rate,
         "null_exact_binomial": null_binom,
         "probe_interaction_chance_bound": probe_bound}, indent=1))
    print(f"[analyze ok] {len(det_rows)} scored cells; "
          f"interaction stop flag = {stop_interaction}")
    return 0


def stage_analyze_confirmatory(base: Path = OUT) -> int:
    """Confirmatory-replication readout (rules commit 29f3a81, fixed before
    any confirmatory run existed): Wilson counts per object, gen-3
    stratified + pooled comparison, exact McNemar D vs TAP, exact binomial
    null check. Thresholds come from the ORIGINAL calibration pool; the
    recomputed original gen-3 counts are asserted against the shipped
    t42_summary.json (same-ruler recompute-assertion). Gates no verdict."""
    cruns = conf_plan()
    missing = [t for (t, *_r) in cruns
               if not (base / "runs" / t / "scores.json").exists()]
    if missing:
        print(f"REFUSING TO ANALYZE (confirmatory): {len(missing)} of "
              f"{len(cruns)} runs missing (first: {missing[:3]})",
              file=sys.stderr)
        return 2
    try:
        q95 = _calib_q95(base, plan())          # ORIGINAL pool, reused
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    def _det_rows(run_list, mode: str):
        """mode='confirmatory': every run must carry the gen-3 marker
        (construction-frozen guard, raises otherwise). mode='original_gen3':
        keep only inj runs of the original list's gen-3 stratum."""
        rows = []
        for (tag, ds, s, kind, cls, rho) in run_list:
            if cls != "interaction":
                continue
            rec = _load_scores(base / "runs" / tag)
            gen3 = "beta_residualisation" in rec["meta"]
            if mode == "confirmatory":
                assert gen3, (f"{tag}: confirmatory run without the gen-3 "
                              f"construction marker")
            elif kind != "inj" or not gen3:
                continue
            t = rec["meta"]["target"]
            for obj in OBJECTS:
                score = rec["target_scores"].get(obj, {}).get(t)
                if score is None:
                    continue
                rows.append({"tag": tag, "object": obj, "dataset": ds,
                             "host_seed": s, "kind": kind, "target": t,
                             "score": float(score),
                             "q95": q95[(obj, ds, t)],
                             "detected": bool(score > q95[(obj, ds, t)])})
        return rows

    conf = _det_rows(cruns, mode="confirmatory")
    orig_g3 = _det_rows(plan(), mode="original_gen3")

    # same-ruler recompute-assertion against the shipped summary
    stored = json.loads((base / "t42_summary.json").read_text()
                        )["interaction_by_generation"]["gen3"][
                            "detected_by_object"]
    for obj in OBJECTS:
        k = sum(r["detected"] for r in orig_g3 if r["object"] == obj)
        assert k == stored[obj], (f"recompute mismatch, original gen-3 "
                                  f"{obj}: {k} != stored {stored[obj]}")

    def _count(rows, obj, kind):
        d = [r["detected"] for r in rows
             if r["object"] == obj and r["kind"] == kind]
        k, n = sum(d), len(d)
        lo, hi = wilson(k, n)
        return {"detected": k, "n": n,
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)}

    def _pairs(rows, kind="inj"):
        by = {}
        for r in rows:
            if r["kind"] != kind or r["object"] not in ("SNI-D", "P"):
                continue
            by.setdefault(r["tag"], {})[r["object"]] = r["detected"]
        return [(v["SNI-D"], v["P"]) for v in by.values()
                if len(v) == 2]

    counts = {obj: {"inj_confirmatory": _count(conf, obj, "inj"),
                    "null_confirmatory": _count(conf, obj, "null"),
                    "inj_original_gen3": _count(orig_g3, obj, "inj"),
                    "inj_pooled_gen3": _count(conf + orig_g3, obj, "inj")}
              for obj in OBJECTS}
    out = {"alpha": ALPHA,
           "rules": "docs/T42_confirmatory_replication_rules.md @ 29f3a81",
           "counts": counts,
           "mcnemar_D_vs_TAP": {
               "confirmatory_6": mcnemar_exact_pairs(_pairs(conf)),
               "pooled_gen3_9": mcnemar_exact_pairs(_pairs(conf + orig_g3))},
           "null_binomial_p_geq_k": {
               obj: round(binom_tail(counts[obj]["null_confirmatory"]
                                     ["detected"], 6, ALPHA), 6)
               for obj in OBJECTS},
           "recompute_assertion_original_gen3":
               "passed vs t42_summary.json"}
    pd.DataFrame(conf).to_csv(base / "t42_confirmatory_detection.csv",
                              index=False)
    (base / "t42_confirmatory.json").write_text(json.dumps(out, indent=1))
    print("[analyze-confirmatory ok] " + "; ".join(
        f"{o}: inj {counts[o]['inj_confirmatory']['detected']}/6, "
        f"pooled gen3 {counts[o]['inj_pooled_gen3']['detected']}/9"
        for o in ("SNI-D", "P")))
    return 0


# --------------------------------------------------------------------------- #
# selftest -- known answers, exact expectations
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    # -- constructions on a fixture table with known structure ------------- #
    rng = np.random.default_rng(0)
    n = 4000
    c1 = rng.standard_normal(n)
    t = 0.7 * c1 + 0.3 * rng.standard_normal(n)     # c1: strong partner
    c2 = rng.standard_normal(n)                     # independent partner
    p = make_proxy("exact", None, t, None, rng)
    check(_corr(p, t) > 0.9999, "exact: corr = 1")
    for rho in (0.95, 0.80, 0.60):
        p = make_proxy("noisy", rho, t, None, rng)
        check(abs(_corr(p, t) - rho) < 0.05, f"noisy{int(rho*100)}: corr ~ rho")
    p = make_proxy("interaction", None, t, c2, rng)
    check(abs(_corr(p, t)) < 0.10, "interaction: marginally ~independent")
    sgn = np.where(_z(c2) >= 0, 1.0, -1.0)
    check(_corr(p * sgn, t) > 0.5, "interaction: jointly predictive with q")
    p = make_proxy("consequence", None, t, None, rng)
    check(0.30 < _corr(p, t) < 0.98, "consequence: thresholded, correlated")
    p = make_proxy("decoy", None, t, c1, rng)
    zq = _z(c1)
    rp = p - np.polyval(np.polyfit(zq, p, 1), zq)
    rt = t - np.polyval(np.polyfit(zq, t, 1), zq)
    check(abs(_corr(p, t)) > 0.10 and abs(_corr(rp, rt)) < 0.10,
          "decoy: correlated, zero incremental")
    # A t-correlated partner keeps marginal independence (parity: the bias
    # E[z_t^2 sign(z_q)] is odd in z_q) but collapses the proxy toward
    # |z_q| -- a partner-only function. The non-degeneracy assertion, not
    # the marginal one, must catch it.
    try:
        make_proxy("interaction", None, t, c1, rng)   # correlated partner
        check(False, "interaction with correlated partner must be refused")
    except AssertionError as e:
        check("degenerates" in str(e),
              "correlated partner refused by the non-degeneracy guard")
    # Skewed real-world shapes (the 2026-08-25 field failure): with a
    # median split the parity argument needs no symmetry -- both t and q
    # heavily skewed must still yield a marginally independent proxy.
    t_sk = rng.exponential(1.0, n)
    q_sk = rng.exponential(1.0, n)
    p = make_proxy("interaction", None, t_sk, q_sk, rng)
    check(abs(_corr(p, t_sk)) < 0.10,
          "interaction on skewed t,q: marginal independence holds")
    sgn_sk = np.where(_z(q_sk) > float(np.median(_z(q_sk))), 1.0, -1.0)
    check(_corr(p * sgn_sk, t_sk) > 0.5,
          "interaction on skewed t,q: jointly predictive")
    # Tie-heavy partner: the median split cannot balance -> the balance
    # guard must refuse (and run_case then falls back to the next partner).
    q_tie = np.zeros(n)
    q_tie[: n // 10] = rng.standard_normal(n // 10)
    try:
        make_proxy("interaction", None, t, q_tie, rng)
        check(False, "tie-heavy partner must be refused")
    except AssertionError as e:
        check("imbalanced" in str(e),
              "tie-heavy partner refused by the balance guard")
    # Heteroskedastic coupling (the MIMIC_s3 field failure): t's variance
    # depends on q's median split while linear corr(t, q) ~ 0. The
    # residualized construction must pass where the raw one cannot.
    q_h = rng.standard_normal(n)
    t_h = rng.standard_normal(n) * (1.0 + 0.8 * (q_h > np.median(q_h)))
    check(abs(_corr(t_h, q_h)) < 0.1, "hetero fixture: linear corr ~ 0")
    p = make_proxy("interaction", None, t_h, q_h, rng)
    check(abs(_corr(p, t_h)) < 0.10,
          "interaction under heteroskedastic coupling: marginal holds")
    sgn_h = np.where(_z(q_h) > float(np.median(_z(q_h))), 1.0, -1.0)
    check(_corr(p * sgn_h, t_h) > 0.5,
          "interaction under heteroskedastic coupling: jointly predictive")

    # -- P4K-B sentinel: OR trigger + self-calibrated baseline ------------- #
    import tempfile
    check(_is_slow(700, 500, None) == [],
          "sentinel: under archived, no self-baseline -> not slow")
    check(_is_slow(750.0, 500, None) == [],
          "sentinel: exactly 1.5x archived is NOT slow (strict >)")
    check(_is_slow(1000, 500, None) == ["archived(500s)"],
          "sentinel: archived breach named")
    check(_is_slow(400, 500, 200) == ["self(200s)"],
          "sentinel OR: below archived but above self-baseline -> slow")
    check(_is_slow(800, 500, 200) == ["archived(500s)", "self(200s)"],
          "sentinel OR: both conditions named")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) \
            as td2:
        rd = Path(td2)
        for tag, wall, mt in [("r1", 100.0, 1000), ("r2", 300.0, 2000),
                              ("r3", 200.0, 3000), ("r4", 9999.0, 4000)]:
            d = rd / tag
            d.mkdir()
            (d / "scores.json").write_text(json.dumps(
                {"meta": {"tag": tag, "dataset": "MIMIC",
                          "wall_train_sec": wall}}))
            os.utime(d / "scores.json", (mt, mt))
        sb = _self_baseline("MIMIC", rd)
        check(sb["median_sec"] == 200.0
              and sb["source_runs"] == ["r1", "r2", "r3"],
              "self-baseline: median of FIRST 3 by completion time, "
              "later run excluded")
        check(_self_baseline("eICU", rd) is None,
              "self-baseline: <3 samples -> None (archived-only regime)")

    # -- analyzer on a synthetic run set with known detections ------------- #
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) \
            as td:
        tmp = Path(td)
        (tmp / "runs").mkdir()
        crng = np.random.default_rng(7)
        for (tag, ds, s, kind, cls, rho) in plan():
            tgt = "T0"
            if kind == "calib":
                ts = {obj: {tgt: float(crng.uniform(0, 1))}
                      for obj in OBJECTS}
            else:
                v = {"SNI-D": 2.0, "P": 2.0}          # always above q95
                if cls == "interaction":
                    # known counts: D detects seeds {1,2,3} on MIMIC only if
                    # seed <= 3 -> craft D 3/6 (MIMIC only), P 1/6 (one cell)
                    v["SNI-D"] = 2.0 if ds == "MIMIC" else -1.0
                    v["P"] = 2.0 if (ds, s) == ("MIMIC", 1) else -1.0
                ts = {}
                for obj in OBJECTS:
                    base_v = v.get(obj, -1.0 if obj != "SNI-D" else 2.0)
                    ts[obj] = {tgt: base_v}
                # Permutation-on-SNI: exactly at the quantile -> NOT detected
                # (strict >). Build after quantile known? use huge negative
                # placeholder; replaced below by exact-quantile pass 2.
                ts["Permutation-on-SNI"] = {tgt: None}
            rd = tmp / "runs" / tag
            rd.mkdir()
            rd.joinpath("scores.json").write_text(json.dumps(
                {"meta": {"target": tgt, "condition":
                          _cond_name(cls, rho) if cls else "calib"},
                 "target_scores": {o: d for o, d in ts.items()
                                   if d.get(tgt) is not None},
                 "row_ranks": {o: {tgt: 1} for o in OBJECTS}}))
        # pass 2: set Permutation-on-SNI injection scores to the exact q95
        for ds in DATASETS:
            vals = []
            for (tag, d2, s, kind, *_x) in plan():
                if d2 == ds and kind == "calib":
                    rec = json.loads(
                        (tmp / "runs" / tag / "scores.json").read_text())
                    vals.append(rec["target_scores"]["Permutation-on-SNI"]["T0"])
            q = float(np.quantile(vals, 0.95))
            for (tag, d2, s, kind, *_x) in plan():
                if d2 == ds and kind != "calib":
                    fp = tmp / "runs" / tag / "scores.json"
                    rec = json.loads(fp.read_text())
                    rec["target_scores"]["Permutation-on-SNI"] = {"T0": q}
                    fp.write_text(json.dumps(rec))
        # P4N SS2.1 fixture: exactly one interaction injection carries a
        # beta field -> must be classified gen3, the rest gen2.
        for (tag, d2, s, kind, cls, rho) in plan():
            if cls == "interaction" and kind == "inj" \
                    and (d2, s) == ("MIMIC", 1):
                fp = tmp / "runs" / tag / "scores.json"
                rec = json.loads(fp.read_text())
                rec["meta"]["beta_residualisation"] = 0.123
                fp.write_text(json.dumps(rec))
        rc = stage_analyze(base=tmp)
        check(rc == 0, "analyzer runs on complete fixture set")
        summ = json.loads((tmp / "t42_summary.json").read_text())
        cnt = {(r["object"], r["condition"], r["kind"]): r["detected"]
               for r in summ["counts"]}
        check(cnt[("SNI-D", "exact", "inj")] == 6, "SNI-D exact: 6/6 detected")
        check(cnt[("Permutation-on-SNI", "exact", "inj")] == 0,
              "score == q95 is NOT a detection (strict >)")
        check(summ["interaction_counts"]["SNI-D"] == 3
              and summ["interaction_counts"]["P"] == 1,
              "interaction counts crafted 3 and 1")
        check(summ["stop_report_interaction_D_wins"] is True,
              "stop flag fires at D=3/6, P=1/6")
        bg = summ["interaction_by_generation"]
        check(bg["gen3"]["n_by_dataset"] == {"MIMIC": 1}
              and bg["gen2"]["n_by_dataset"] == {"MIMIC": 2, "eICU": 3},
              "generation decomposition: 1 gen3 (MIMIC), 5 gen2")
        check(bg["gen3"]["detected_by_object"].get("SNI-D", 0) == 1,
              "gen3 detection count carried per object")
        r36 = [r for r in summ["counts"]
               if r["object"] == "SNI-D" and r["condition"] == "interaction"
               and r["kind"] == "inj"][0]
        check(abs(r36["wilson_lo"] - 0.1876) < 1e-3
              and abs(r36["wilson_hi"] - 0.8124) < 1e-3,
              "Wilson CI for 3/6 = (0.1876, 0.8124), hand-computed")
        check(not any("pooled" in str(k) for k in summ),
              "no pooled-across-classes field exists")
        # flip: P detects 2 -> stop flag must NOT fire
        for (tag, d2, s, kind, cls, rho) in plan():
            if cls == "interaction" and kind == "inj" \
                    and (d2, s) == ("MIMIC", 2):
                fp = tmp / "runs" / tag / "scores.json"
                rec = json.loads(fp.read_text())
                rec["target_scores"]["P"] = {"T0": 2.0}
                fp.write_text(json.dumps(rec))
        stage_analyze(base=tmp)
        summ = json.loads((tmp / "t42_summary.json").read_text())
        check(summ["stop_report_interaction_D_wins"] is False,
              "stop flag off at P=2/6")
        # incompleteness refusal
        some = next((tmp / "runs").iterdir())
        (some / "scores.json").rename(some / "scores.json.bak")
        check(stage_analyze(base=tmp) == 2, "missing run refuses analyze")

    # -- confirmatory replication machinery (rules 29f3a81) ---------------- #
    check([_seed_index(s) for s in [1, 2, 3, 5, 8, 13]] == [0, 1, 2, 3, 4, 5],
          "seed index: originals 0-2 unchanged, confirmatory 3-5")
    check([_seed_index(s) % 3 for s in [5, 8, 13]] == [0, 1, 2],
          "confirmatory target rotation repeats the original ranks")
    cp = conf_plan()
    check(len(cp) == 12 and all(c == "interaction" for (_t, _d, _s, _k, c, _r)
                                in cp), "conf_plan: 12 interaction runs")
    check(not ({t for (t, *_r) in cp} & {t for (t, *_r) in plan()}),
          "conf_plan tags disjoint from the original plan")
    m = mcnemar_exact_pairs([(False, True)] * 3 + [(True, True),
                                                   (False, False)])
    check(m == {"n_pairs": 5, "D_only": 0, "TAP_only": 3,
                "p_two_sided": 0.25},
          f"mcnemar: 0 vs 3 discordant -> p = 2/8 = 0.25 ({m})")
    m = mcnemar_exact_pairs([(True, False), (False, True)])
    check(m["p_two_sided"] == 1.0, "mcnemar: 1 vs 1 -> p clamps to 1")
    check(mcnemar_exact_pairs([(True, True)])["p_two_sided"] == 1.0,
          "mcnemar: no discordant pairs -> p = 1")
    check(abs(binom_tail(1, 6, 0.05) - (1 - 0.95 ** 6)) < 1e-12
          and abs(binom_tail(0, 6, 0.05) - 1.0) < 1e-12,
          "binom tail: k=1 matches 1-(0.95^6); k=0 is 1")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["plan", "run", "analyze", "selftest",
                             "run-confirmatory", "analyze-confirmatory"])
    ap.add_argument("--slice", default="0/1")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "plan":
        return stage_plan()
    if a.stage == "run":
        return stage_run(a.slice)
    if a.stage == "run-confirmatory":
        return stage_run(a.slice, planfn=conf_plan)
    if a.stage == "analyze-confirmatory":
        return stage_analyze_confirmatory()
    return stage_analyze()


if __name__ == "__main__":
    raise SystemExit(main())
