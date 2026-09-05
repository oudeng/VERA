"""T2.5 — the R2-1 pilot: does SNI-D survive contact with post-hoc explainers?

R2-1 is the one reviewer point no amount of rewriting can answer, and it decides
the paper's positioning. The go/no-go rule was written and committed **before any
run** in `docs/T25_pilot_decision_rule.md`; this file only produces the numbers
the rule consumes.

Four comparators, all producing a d x d directed matrix with row = target,
col = source (the convention of both `SNIImputer.compute_dependency_matrix`
(`imputer.py:395`) and the synthetic ground truth (`synth_generate_s5.py:8`)):

  1. **SNI-D**            — free at training time
  2. **MissForest importances** — `models_[col].feature_importances_`, i.e. the
     forests MissForest itself fitted. The most isomorphic opponent.
  3. **SHAP-on-MissForest**     — TreeSHAP on those same forests
  4. **Permutation-on-MissForest** — permutation importance on those same forests

Comparators 2-4 deliberately share MissForest's own forests, so they differ only
in how importance is read out of one model. Fitting a separate surrogate per
method would confound "which readout" with "which model".

Data: R0's own synthetic set (`project_sni_R0/sni/data/synth_s5`, read-only) --
3 regimes x 5 seeds, 2000 x 12, MAR@30 %, with the true DAG shipped alongside.
Reusing it keeps the comparison with R0's Table III honest and generates nothing.

Two deliberate departures, both stated because neither is free:

* **Hyperparameters are the main-experiment ones** (200 epochs, 16 heads,
  emb 128, hidden [256,128,64], early stopping off). R0's synthetic run used
  epochs 50 / heads 4 / emb 32, so its "Prior-Only wins" result may be
  under-training -- that is B18, and inheriting it would defeat the purpose.
* **SNI runs on CPU, not CUDA.** P2e §1.2 asks for CUDA "for comparability with
  the existing smoke/probe data", but there is no prior SNI run on these
  synthetic tables, so there is nothing to be comparable with; meanwhile §3.1 of
  the same instruction makes CPU the canonical R1 device. Running CUDA here would
  buy nothing and force a re-run under §3. Metrics are device-dependent (B83), so
  this is recorded rather than glossed.

All four matrices are row-normalized before scoring. D is row-normalized by
construction and `feature_importances_` already sums to 1 per row; SHAP and
permutation values are not, and leaving them unnormalised would let a method with
a few large rows dominate a matrix-wide ranking.

    env PYTHONHASHSEED=2025 python experiments/pilot_r21.py --smoke
    env PYTHONHASHSEED=2025 python experiments/pilot_r21.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))     # runnable as a script, not just -m

#: P7-A closeout: no private absolute path in a published file. The
#: R0 tree is not in this repository (it holds restricted derived
#: tables); point at it with SNI_R0_ROOT, and default to the sibling
#: directory a full checkout would have. A clone that lacks it gets a
#: path it can act on rather than a stranger's home directory.
R0_ROOT = Path(os.environ.get("SNI_R0_ROOT",
                    Path(__file__).resolve().parents[2]
                    / "project_sni_R0"))
SYNTH = R0_ROOT / "sni" / "data" / "synth_s5"
OUT = CODE_ROOT / "results" / "T2.5_pilot"

REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SEEDS = [2025, 2026, 2027, 2028, 2029]
STEM = "synth_{regime}_n2000_d12_seed{seed}"

METHODS = ["SNI-D", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_cell(regime: str, seed: int):
    stem = STEM.format(regime=regime, seed=seed)
    complete = pd.read_csv(SYNTH / f"{stem}_complete.csv")
    missing = pd.read_csv(SYNTH / f"{stem}_MAR_30per.csv")
    mask = pd.read_csv(SYNTH / f"{stem}_MAR_30per_mask.csv")
    G = pd.read_csv(SYNTH / f"{stem}_ground_truth_G.csv", index_col=0)
    meta = json.loads((SYNTH / f"{stem}_metadata.json").read_text())
    # The types live under "data", not at the top level. Reading them from the
    # top level silently yields an empty categorical list, which would send c0/c1
    # -- integer categoricals -- down the continuous path in every method.
    blk = meta["data"]
    cat = [c for c in blk["cat_vars"] if c in complete.columns]
    cont = [c for c in blk["cont_vars"] if c in complete.columns]
    missing_cols = set(complete.columns) - set(cat) - set(cont)
    if missing_cols:
        raise ValueError(f"{stem}: columns typed by neither list: {missing_cols}")

    # Align the mask to the table's own column order and coerce to bool. The CSV
    # column order (c0, c1, x0..x9) is not metadata's all_vars order (x0..x9, c0,
    # c1), so everything is reindexed to the table rather than assumed aligned.
    mask = mask[list(complete.columns)].astype(bool)
    G = G.reindex(index=list(complete.columns), columns=list(complete.columns))

    # Cross-check G against the independently recorded parent lists. Two
    # representations of the same truth should agree; if they do not, scoring
    # against either is meaningless.
    #
    # Only two of the three regimes can be checked this way. linear_gaussian and
    # nonlinear_mixed come from `synth_generate_s5.py`, whose metadata carries
    # `parents_named`/`parents_index`; interaction_xor comes from
    # `sanity_check_v2_s5.py:502`, which writes ground_truth_G.csv but not the
    # parent lists. Its truth is therefore taken on trust from a single file --
    # recorded here so the asymmetry is visible in the report rather than
    # discovered later.
    checked = "parents_named" in blk
    if checked:
        for target, parents in blk["parents_named"].items():
            row = G.loc[target]
            got = sorted(row[row > 0].index.tolist())
            if got != sorted(parents):
                raise ValueError(f"{stem}: G row {target} = {got} but metadata "
                                 f"says parents_named = {sorted(parents)}")
    return complete, missing, mask, G, cat, cont, checked


def row_normalise(M: pd.DataFrame) -> pd.DataFrame:
    A = np.abs(M.to_numpy(dtype=float))
    np.fill_diagonal(A, 0.0)
    s = A.sum(axis=1, keepdims=True)
    A = np.divide(A, s, out=np.zeros_like(A), where=s > 0)
    return pd.DataFrame(A, index=M.index, columns=M.columns)


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def run_sni(missing, mask, cat, cont, seed, cols):
    from sni.imputer import SNIConfig, SNIImputer
    from common import determinism
    import yaml
    proto = yaml.safe_load(
        (CODE_ROOT / "configs" / "training_protocol.yaml").read_text())["protocol"]
    epochs = int(proto["epochs"]["SNI"])

    determinism.apply("deterministic", seed=seed)
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.epochs = epochs
    imp.cfg.early_stopping_patience = epochs + 1        # protocol: no early stop

    t0 = time.time()
    imp.impute(X_missing=missing, X_complete=None, mask_df=mask[missing.columns])
    impute_sec = time.time() - t0

    t1 = time.time()
    D = imp.compute_dependency_matrix()
    audit_sec = time.time() - t1
    D = D.reindex(index=cols, columns=cols).fillna(0.0)
    return {"SNI-D": (D, impute_sec, audit_sec)}


def run_missforest_family(missing, cat, cont, seed, cols, readouts=None):
    """One MissForest fit; three readouts of its own per-column forests.

    `readouts` selects a SUBSET of the three (default: all three, the
    original path unchanged). Each readout is a self-contained function of
    the fitted forest -- ``feature_importances_`` is a model attribute,
    TreeSHAP is deterministic, and ``permutation_importance`` seeds its own
    RNG from ``random_state`` -- so computing a subset leaves the computed
    values bit-identical. Used by the A3 cost probe to measure each
    readout's peak memory in its own process (P5R-H SS3 / third review
    SS7.3); no scientific artifact is generated through this path.
    """
    import shap
    from sklearn.inspection import permutation_importance
    from baselines.registry import build_baseline_imputer
    from baselines.schema import DataSchema
    from common import determinism

    determinism.apply("deterministic", seed=seed)
    imp = build_baseline_imputer("MissForest", categorical_vars=cat,
                                 continuous_vars=cont, seed=seed, use_gpu=False)
    schema = DataSchema(categorical_vars=cat, continuous_vars=cont)
    t0 = time.time()
    completed = imp.impute(missing, schema)
    impute_sec = time.time() - t0

    impl = getattr(imp, "_impl", imp)
    models = getattr(impl, "models_", {}) or {}

    # The forests were fitted on integer-coded categoricals (`MissForest_v2.py:195`
    # codes them, `:449` decodes back to labels), so the frame MissForest *returns*
    # is not the frame its models accept -- TreeSHAP fails outright on the string
    # labels. Rebuild the numeric view the models saw.
    numeric = completed.copy()
    for c in cat:
        if str(numeric[c].dtype) == "category":
            numeric[c] = numeric[c].cat.codes.astype(float)
        else:
            numeric[c] = pd.Categorical(numeric[c]).codes.astype(float)

    # Verify the coding rather than trusting it. What has to hold is that
    # codes <-> labels is the *same bijection* MissForest used: it built
    # `cats = list(df[col].cat.categories)` and decoded with `cats[int(code)]`
    # (`MissForest_v2.py:194`, `:449`). A round trip through the returned frame's
    # own category index tests exactly that, element by element.
    #
    # NOT tested by replaying a forest on this frame and expecting MissForest's
    # own imputed values back: MissForest is iterative, `models_[col]` is only the
    # final round's forest, and the values it wrote were predicted from
    # `df_current` as it stood mid-round -- later columns in the same round, and
    # the same column in later rounds, moved afterwards. An earlier version of
    # this check asserted that equality and fired at max |diff| 6.5e-02 on x1,
    # which was the premise being wrong, not the coding.
    for c in cat:
        if str(completed[c].dtype) != "category":
            raise RuntimeError(f"{c}: MissForest returned dtype "
                               f"{completed[c].dtype}, not category; the code "
                               f"mapping cannot be recovered from it")
        cats = list(completed[c].cat.categories)
        codes = numeric[c].to_numpy(dtype=int)
        if codes.min() < 0 or codes.max() >= len(cats):
            raise RuntimeError(f"{c}: codes out of range [0, {len(cats) - 1}]; "
                               f"got [{codes.min()}, {codes.max()}] — a -1 means "
                               f"an unmapped value was silently coded as missing")
        back = pd.Series([cats[i] for i in codes], index=completed.index)
        if not back.astype(str).equals(completed[c].astype(str)):
            n_bad = int((back.astype(str) != completed[c].astype(str)).sum())
            raise RuntimeError(f"{c}: code round trip disagrees with the returned "
                               f"labels in {n_bad} rows. Refusing to compute SHAP "
                               f"or permutation importance on a frame the models "
                               f"did not see.")

    # Column ordering is the other half of the contract: each forest was fitted on
    # `[c for c in columns if c != target]`, so its feature count must match.
    for target, model in models.items():
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and n_in != len(cols) - 1:
            raise RuntimeError(f"{target}: forest expects {n_in} features but the "
                               f"table has {len(cols) - 1} others; the feature "
                               f"ordering assumption is wrong")
    d = len(cols)
    ALL_READOUTS = ("MissForest-importance", "SHAP-on-MissForest",
                    "Permutation-on-MissForest")
    want = tuple(readouts) if readouts else ALL_READOUTS
    unknown = [r for r in want if r not in ALL_READOUTS]
    if unknown:
        raise ValueError(f"unknown readout(s) {unknown}; expected "
                         f"{list(ALL_READOUTS)}")
    mats = {k: pd.DataFrame(np.full((d, d), np.nan), index=cols, columns=cols)
            for k in want}
    audit = {k: 0.0 for k in want}
    no_model = []

    for target in cols:
        model = models.get(target)
        if model is None:
            # A column MissForest never had to impute has no forest. Recorded and
            # left NaN rather than filled with zeros, which would silently score
            # as "no dependencies" instead of "not measured".
            no_model.append(target)
            continue
        others = [c for c in completed.columns if c != target]
        X = numeric[others]                  # the coding the forests were fitted on
        y = numeric[target]

        if "MissForest-importance" in want:
            t = time.time()
            fi = np.asarray(model.feature_importances_, dtype=float)
            audit["MissForest-importance"] += time.time() - t
            mats["MissForest-importance"].loc[target, others] = fi

        if "SHAP-on-MissForest" in want:
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ex = shap.TreeExplainer(model)
                sv = ex.shap_values(X, check_additivity=False)
            sv = np.asarray(sv)
            # Classifiers give one array per class; average |value| over classes.
            while sv.ndim > 2:
                sv = np.abs(sv).mean(axis=-1)
            mats["SHAP-on-MissForest"].loc[target, others] = np.abs(sv).mean(axis=0)
            audit["SHAP-on-MissForest"] += time.time() - t

        if "Permutation-on-MissForest" in want:
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pi = permutation_importance(model, X, y, n_repeats=5,
                                            random_state=seed, n_jobs=1)
            mats["Permutation-on-MissForest"].loc[target, others] = pi.importances_mean
            audit["Permutation-on-MissForest"] += time.time() - t

    return ({k: (mats[k], impute_sec, audit[k]) for k in mats}, no_model)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def measured_rows(M: pd.DataFrame) -> np.ndarray:
    """Rows for which a method produced any estimate at all.

    All-NaN and all-zero both count as unmeasured. The zero case is not
    hypothetical: `compute_dependency_matrix` leaves a row of zeros for any target
    absent from `attention_maps` (`imputer.py:406`), and this file fills SNI's
    reindexed NaNs with 0. Scored naively that row reads as "predicted no
    dependencies" -- a confident wrong answer -- instead of "did not answer".
    """
    A = M.to_numpy(dtype=float)
    return ~(np.isnan(A) | (A == 0.0)).all(axis=1)


def score(M: pd.DataFrame, G: pd.DataFrame, keep: np.ndarray = None) -> dict:
    """Axis A. Off-diagonal only, restricted to `keep` rows.

    `keep` is the intersection of rows every method could estimate, not each
    method's own coverage. MissForest fits no forest for a fully observed column
    (x0 is the MAR driver), so it yields 11 rows where SNI-D yields 12 -- and x0's
    row is all negatives, which shifts AUROC on its own. Scoring each method on
    its own row set would compare them on different problems.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    A = row_normalise(M).to_numpy(dtype=float)
    T = G.to_numpy(dtype=float)
    if keep is None:
        keep = measured_rows(M)
    off = ~np.eye(len(A), dtype=bool)
    sel = off & keep[:, None]

    y = (T[sel] > 0).astype(int)
    s = A[sel]
    if y.sum() == 0 or y.sum() == y.size:
        return {"auroc": np.nan, "auprc": np.nan, "prec_at_k": np.nan,
                "shd": np.nan, "n_true_edges": int(y.sum()),
                "n_rows_scored": int(keep.sum())}

    k = int(y.sum())
    order = np.argsort(-s)
    topk = np.zeros_like(y)
    topk[order[:k]] = 1
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "prec_at_k": float(y[order[:k]].mean()),
            # SHD with a top-K binarisation: both directions of mismatch counted.
            "shd": int(np.sum(topk != y)),
            "n_true_edges": k, "n_rows_scored": int(keep.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--smoke", action="store_true",
                    help="one regime, one seed -- validates the harness cheaply")
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2

    regimes = a.regimes[:1] if a.smoke else a.regimes
    seeds = a.seeds[:1] if a.smoke else a.seeds
    OUT.mkdir(parents=True, exist_ok=True)

    rows, mats_out = [], {}
    for regime in regimes:
        for seed in seeds:
            complete, missing, mask, G, cat, cont, gt_checked = load_cell(regime, seed)
            cols = list(complete.columns)
            got = {}
            got.update(run_sni(missing, mask, cat, cont, seed, cols))
            fam, no_model = run_missforest_family(missing, cat, cont, seed, cols)
            got.update(fam)
            if no_model:
                print(f"  note: no MissForest forest for {no_model} "
                      f"({regime}/s{seed}) -- rows left unmeasured", flush=True)

            # One row set for every method in this cell: the intersection of
            # what each could estimate.
            common = np.ones(len(cols), dtype=bool)
            for M, _, _ in got.values():
                common &= measured_rows(M)

            for name, (M, imp_s, aud_s) in got.items():
                sc = score(M, G, keep=common)
                sc["own_rows"] = int(measured_rows(M).sum())
                rows.append({"regime": regime, "seed": seed, "method": name,
                             "gt_cross_checked": gt_checked,
                             "impute_sec": round(imp_s, 2),
                             "audit_sec": round(aud_s, 2),
                             "total_sec": round(imp_s + aud_s, 2), **sc})
                mats_out[f"{regime}|{seed}|{name}"] = row_normalise(M)
                # Persist per cell, not at the end. B79 cost 42 of 49 rows by
                # holding results until a final write, and here the D matrices --
                # the only input to axis B -- would be lost with them. A run that
                # dies at cell 14 must not cost the first 13.
                row_normalise(M).to_csv(
                    OUT / f"D_{regime}_s{seed}_{name.replace('/', '-')}.csv")
                print(f"[{regime[:14]:<14} s{seed} {name:<26}] "
                      f"AUROC={sc['auroc']:.4f} AUPRC={sc['auprc']:.4f} "
                      f"P@K={sc['prec_at_k']:.4f} SHD={sc['shd']:<3} "
                      f"total={imp_s + aud_s:7.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"{a.tag}_cells.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{a.tag}_cells.csv", index=False)   # final, idempotent
    print(f"\nwrote {OUT}/{a.tag}_cells.csv ({len(df)} rows) and "
          f"{len(mats_out)} matrices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
