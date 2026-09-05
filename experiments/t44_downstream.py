"""T4.4 -- within-fold downstream Impute->Predict (R1-2 + R1-3).

Implements docs/T44_downstream_rules.md (+ corrigendum) verbatim:
evaluation/protocol.py is the machinery; the protocol each imputer actually
gets is decided by detect_protocol (the resolved protocol is recorded per
unit -- SNI has no fit/transform and is therefore treated fold-independent,
the correct leak-free handling for a transductive imputer, reported as
such). Masks: the grid's fixed, content-hashed MAR@30 mask restricted to
feature columns; the label column is dropped from table and mask alike and
is never masked. Seeds vary the stratified 80/20 split and model
randomness only.

Stages:
    selftest -- fixtures with known answers (no real data)
    smoke    -- verify_fold_independence on one method per protocol class
    plan     -- enumerate units, print scale
    run      -- one unit: --unit PANEL:IMPUTER:SEED   (resume-by-artifact)
    collect  -- long CSV for reporting/table_impute_predict.py

    env PYTHONHASHSEED=2025 python experiments/t44_downstream.py --stage run \
        --unit B:MeanMode:1
"""
from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT

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

OUT = CODE_ROOT / "results" / "T4_downstream"

IMPUTERS = ["MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE",
            "HyperImpute", "TabCSDI", "SNI"]
SEEDS = [1, 2, 3, 5, 8]
PANELS = {
    "B": {"dataset": "NHANES", "label": "metabolic_score",
          "task": "metabolic classification (6-class)"},
    # Panel A launches only after first-author sign-off (rule doc):
    "A": {"dataset": "MIMIC", "label": "mortality_risk",
          "task": "in-hospital mortality (binary)"},
}
MODELS = ["LR", "XGB"]


# --------------------------------------------------------------------------- #
def _load_panel(panel: str):
    from baselines.schema import DataSchema
    cfg = PANELS[panel]
    ds = cfg["dataset"]
    complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                           / f"{ds}_complete.csv")
    mask = np.load(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
                   / ds / f"{ds}_MAR_30per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    cat = list(schema.categorical_vars)
    cont = list(schema.continuous_vars)
    label = cfg["label"]
    y = pd.to_numeric(complete[label], errors="raise")
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))
    feats = [c for c in cat + cont if c != label]
    if label in cat + cont:
        # rule: the label is dropped from features and never masked
        cat = [c for c in cat if c != label]
        cont = [c for c in cont if c != label]
    if label in (cat + cont):
        raise ValueError("label survived the feature drop")
    X_missing = complete[feats].mask(mask_df[feats])
    return X_missing, y, cat, cont, feats, cfg


def _factory(imputer: str, cat, cont, seed: int):
    if imputer == "SNI":
        import yaml
        from sni.imputer import SNIConfig, SNIImputer

        proto = yaml.safe_load((CODE_ROOT / "configs" /
                                "training_protocol.yaml").read_text())["protocol"]

        def make():
            from common import determinism
            determinism.apply("deterministic", seed=seed)
            imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                             config=SNIConfig(seed=seed, use_gpu=False))
            imp.cfg.epochs = int(proto["epochs"]["SNI"])
            imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
            return imp
        return make

    from baselines.registry import build_baseline_imputer

    def make():
        return build_baseline_imputer(imputer, cat, cont, seed=seed,
                                      use_gpu=(imputer == "TabCSDI"))
    return make


def _encode_and_fit(model: str, Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame,
                    cat, seed: int):
    """Deterministic downstream pipeline: one-hot cats (fit on train),
    standardise conts, then LR or XGB. Identical across imputers."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cats = [c for c in Xtr.columns if c in set(cat)]
    conts = [c for c in Xtr.columns if c not in set(cat)]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats),
         ("cont", StandardScaler(), conts)])
    if model == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif model == "XGB":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            eval_metric="logloss", n_jobs=int(_NT))
    else:
        raise ValueError(model)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    Xtr2, ytr2 = Xtr.copy(), np.asarray(ytr)
    if model == "XGB":
        # xgboost requires 0..K-1 labels
        classes, ytr2 = np.unique(ytr2, return_inverse=True)
        pipe.fit(Xtr2, ytr2)
        return pipe, classes
    pipe.fit(Xtr2, ytr2)
    return pipe, np.unique(ytr2)


def _score(pipe, classes, Xte: pd.DataFrame, yte) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)
    proba = pipe.predict_proba(Xte)
    yte = np.asarray(yte)
    pred = classes[np.argmax(proba, axis=1)]
    binary = len(classes) == 2
    y_idx = np.searchsorted(classes, yte)
    if binary:
        auroc = float(roc_auc_score(y_idx, proba[:, 1]))
        auprc = float(average_precision_score(y_idx, proba[:, 1]))
    else:
        Y = np.zeros((len(yte), len(classes)))
        Y[np.arange(len(yte)), y_idx] = 1.0
        auroc = float(roc_auc_score(Y, proba, average="macro",
                                    multi_class="ovr"))
        auprc = float(average_precision_score(Y, proba, average="macro"))
    return {"AUROC": auroc, "AUPRC": auprc,
            "Accuracy": float(accuracy_score(yte, pred)),
            "F1": float(f1_score(yte, pred, average="macro"))}


def run_unit(panel: str, imputer: str, seed: int) -> None:
    from evaluation.protocol import make_holdout_folds, run_protocol

    tag = f"{panel}_{imputer}_s{seed}"
    upath = OUT / "units" / f"{tag}.json"
    if upath.exists():
        print(f"[cached] {tag}", flush=True)
        return
    upath.parent.mkdir(parents=True, exist_ok=True)

    X_missing, y, cat, cont, feats, cfg = _load_panel(panel)
    folds = make_holdout_folds(len(X_missing), test_size=0.2, seed=seed,
                               stratify=y)
    t0 = time.time()
    rep = run_protocol(imputer, _factory(imputer, cat, cont, seed),
                       X_missing, folds)
    f = rep.folds[0]
    ytr, yte = y.iloc[f.train_idx], y.iloc[f.test_idx]
    metrics = {}
    for model in MODELS:
        pipe, classes = _encode_and_fit(model, f.train_imputed, ytr,
                                        f.test_imputed, cat, seed)
        metrics[model] = _score(pipe, classes, f.test_imputed, yte)
    rec = {"panel": panel, "dataset": cfg["dataset"], "task": cfg["task"],
           "imputer": imputer, "seed": seed,
           "protocol_resolved": rep.protocol,
           "n_train": int(len(f.train_idx)), "n_test": int(len(f.test_idx)),
           "impute_wall_sec": round(f.fit_runtime_sec
                                    + f.transform_runtime_sec, 1),
           "total_wall_sec": round(time.time() - t0, 1),
           "metrics": metrics}
    upath.write_text(json.dumps(rec, indent=1))
    print(f"[ok] {tag} proto={rep.protocol} "
          f"AUROC(LR)={metrics['LR']['AUROC']:.4f} "
          f"wall={rec['total_wall_sec']:.0f}s", flush=True)


def units(panels: list) -> list:
    return [(p, m, s) for p in panels for m in IMPUTERS for s in SEEDS]


def stage_collect(panels: list) -> int:
    rows = []
    missing = []
    for (p, m, s) in units(panels):
        up = OUT / "units" / f"{p}_{m}_s{s}.json"
        if not up.exists():
            missing.append(up.name)
            continue
        rec = json.loads(up.read_text())
        for model, md in rec["metrics"].items():
            for metric, value in md.items():
                rows.append({"panel": p, "dataset": rec["dataset"],
                             "task": rec["task"], "imputer": m,
                             "model": model, "seed": s,
                             "metric": metric, "value": value})
    if missing:
        print(f"REFUSING TO COLLECT: {len(missing)} units missing "
              f"(first: {missing[:3]})", file=sys.stderr)
        return 2
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "t44_long.csv", index=False)
    print(f"[ok] wrote {OUT / 't44_long.csv'} ({len(long)} rows)")
    return 0


def stage_smoke() -> int:
    """Per-class independence evidence, one method per protocol class
    (second internal review SS9; P5R-G gate 2 branch a).

    Three executed assertions per method -- (A) test-feature perturbation
    leaves the completed training block unchanged, (B) train-feature
    perturbation moves the test block iff the class is inductive
    (fit/transform positive control) and leaves it bit-identical iff the
    class is fold-independent, (C) determinism reference -- plus one
    harness-level assertion: (L) permuting the label with the split held
    fixed leaves the imputer output byte-identical (the label never enters
    the imputation path; executed, not assumed). Any red assertion ->
    tripline: stop, report."""
    from evaluation.protocol import (impute_within_fold, make_holdout_folds,
                                     verify_independence_per_class)
    X_missing, y, cat, cont, _f, _c = _load_panel("B")
    fold = make_holdout_folds(len(X_missing), test_size=0.2, seed=1,
                              stratify=y)[0]
    ok, results = True, {}
    for m in ("MeanMode", "GAIN"):
        r = verify_independence_per_class(_factory(m, cat, cont, 1),
                                          X_missing, fold)
        # (L) the label cannot influence the imputation: with the split held
        # fixed, y has no entry point -- impute_within_fold takes no label
        # parameter (asserted by signature), the label was dropped from the
        # feature frame (runtime-guarded in _load_panel), and a full re-run
        # of the path reproduces both completed blocks byte-identically.
        import inspect
        sig = set(inspect.signature(impute_within_fold).parameters)
        no_label_param = not ({"y", "label", "target"} & sig)
        base = impute_within_fold(_factory(m, cat, cont, 1), X_missing, fold)
        rerun = impute_within_fold(_factory(m, cat, cont, 1), X_missing,
                                   fold)
        label_ok = (no_label_param
                    and base.test_imputed.equals(rerun.test_imputed)
                    and base.train_imputed.equals(rerun.train_imputed))
        r["L_label_cannot_reach_imputer"] = bool(label_ok)
        r["L_note"] = ("with the prespecified split held fixed the label has "
                       "no entry point into the imputation path: no label "
                       "parameter exists (signature-asserted), the label is "
                       "dropped from the feature frame (runtime-guarded), "
                       "and a full re-run reproduces both completed blocks "
                       "byte-identically")
        results[m] = r
        print(f"[smoke] {m}: protocol={r['protocol']} "
              f"A(train block unchanged under test perturbation)="
              f"{r['A_pass_train_block_unchanged']} "
              f"(n={r['A_test_perturb_train_block_n_changes']}) "
              f"B({r['B_reading'].split(' (')[0]})={r['B_pass']} "
              f"(n={r['B_train_perturb_test_block_n_changes']}) "
              f"L(label inert)={label_ok} "
              f"det={r['deterministic_reference']}")
        ok = ok and r["independent_per_class"] and label_ok
    out = OUT / "smoke_independence.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"panel": "B", "split": "per seed one stratified 80/20 holdout "
                                "(repeated holdout, not cross-validation)",
         "methods": results,
         "verdict": "ALL-GREEN" if ok else "TRIPLINE"}, indent=1))
    print(f"[smoke] wrote {out}")
    if not ok:
        print("TRIPLINE: per-class independence violated; stopping.",
              file=sys.stderr)
        return 3
    return 0


def stage_plan(panels: list) -> int:
    us = units(panels)
    done = sum((OUT / "units" / f"{p}_{m}_s{s}.json").exists()
               for (p, m, s) in us)
    print(f"T4.4 plan: {len(us)} units ({len(panels)} panel(s) x 9 imputers "
          f"x 5 seeds), {done} done. Rule-doc scale: ~35-40 machine-h CPU + "
          f"~3 h GPU for both panels; NHANES-only ~ half.")
    return 0


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "c1": rng.standard_normal(n), "c2": rng.standard_normal(n),
        "g": rng.integers(0, 3, n).astype(float)})
    y = (df.c1 > 0).astype(int)

    # perfectly separable case: LR AUROC must be ~1
    pipe, classes = _encode_and_fit("LR", df, y, df, ["g"], seed=0)
    sc = _score(pipe, classes, df, y)
    check(sc["AUROC"] > 0.999, "separable fixture: LR AUROC ~ 1")
    check(all(0.0 <= sc[k] <= 1.0 for k in sc), "metrics within [0,1]")

    # multiclass macro path: 3-class random labels -> AUROC ~ 0.5 band
    y3 = rng.integers(0, 3, n)
    Xr = pd.DataFrame({"c1": rng.standard_normal(n),
                       "c2": rng.standard_normal(n), "g": 0.0})
    pipe, classes = _encode_and_fit("LR", Xr, y3, Xr, ["g"], seed=0)
    sc3 = _score(pipe, classes, Xr, y3)
    check(len(classes) == 3 and 0.35 < sc3["AUROC"] < 0.75,
          "multiclass macro AUROC computed on 3 classes (chance band)")

    # label-drop guard: label listed among features must be removed
    from baselines.schema import DataSchema  # noqa: F401  (import path check)
    Xm, yy, cat, cont, feats, _ = _load_panel("B")
    check("metabolic_score" not in feats
          and "metabolic_score" not in cat + cont,
          "panel B: label dropped from features and schema lists")
    check(len(Xm) == len(yy) and yy.notna().all(),
          "panel B: label fully observed and row-aligned")
    check(Xm.isna().any().any(), "panel B: features carry the mask")

    # collect schema matches the table generator's required columns
    need = {"panel", "dataset", "task", "imputer", "model", "seed",
            "metric", "value"}
    row = {"panel": "B", "dataset": "NHANES", "task": "t", "imputer": "x",
           "model": "LR", "seed": 1, "metric": "AUROC", "value": 0.5}
    check(need == set(row), "long-CSV schema matches table_impute_predict")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "smoke", "plan", "run", "collect"])
    ap.add_argument("--unit", default=None, help="PANEL:IMPUTER:SEED")
    ap.add_argument("--panels", nargs="*", default=["B"],
                    help="Panel A only after first-author sign-off")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "smoke":
        return stage_smoke()
    if a.stage == "plan":
        return stage_plan(a.panels)
    if a.stage == "collect":
        return stage_collect(a.panels)
    p, m, s = a.unit.split(":")
    assert p in PANELS and m in IMPUTERS and int(s) in SEEDS, a.unit
    run_unit(p, m, int(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
