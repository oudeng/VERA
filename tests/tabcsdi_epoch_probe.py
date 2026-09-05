"""Does TabCSDI ever use the budget it is given?

The budget sensitivity check (P2b decision 1) raises `epochs` from R0's 200 to
400 and 800 and asks whether the divergence clears. Reading the implementation
first turned up two reasons that question is not quite the one being answered:

* `TabCSDI_v1.py:436-495` early-stops on the **training** loss, patience 20,
  `min_epochs` 50. If it stops at, say, epoch 78, the cap was never binding and
  raising it changes nothing at all.
* `:417` builds the LR schedule as `CosineAnnealingLR(T_max=self.epochs)`. So
  raising the cap also **stretches the annealing**: at epoch 70 a run capped at
  800 sits at a much higher learning rate than one capped at 200. The two arms
  therefore differ in more than budget.

Corroborating evidence that something like this is happening: in the budget run,
NHANES at 4x (800 epochs) finished in 173-235 s while 1x (200 epochs) took
215-226 s. Four times the cap cannot take less time unless training stops early.

This probe counts the epochs actually executed, by wrapping the scheduler's
`step`. Nothing in the imputer is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from baselines.registry import build_baseline_imputer   # noqa: E402
from sni.imputer import SNIConfig, SNIImputer            # noqa: E402
from baselines.schema import DataSchema                 # noqa: E402
from common import determinism                          # noqa: E402

OUT = CODE_ROOT / "results" / "T2b_tabcsdi_budget"


class _EpochCounter:
    """Counts scheduler.step() calls, which the training loop makes once per
    epoch (`TabCSDI_v1.py:484`)."""

    def __init__(self):
        self.n = 0
        self._orig = torch.optim.lr_scheduler.CosineAnnealingLR.step

    def __enter__(self):
        counter = self

        def step(self_, *a, **kw):
            counter.n += 1
            return counter._orig(self_, *a, **kw)

        torch.optim.lr_scheduler.CosineAnnealingLR.step = step
        return self

    def __exit__(self, *exc):
        torch.optim.lr_scheduler.CosineAnnealingLR.step = self._orig
        return False


def probe(dataset: str, seed: int, epochs: int, table_dir: Path, mask_dir: Path,
          disable_early_stop: bool, method: str = "TabCSDI") -> dict:
    complete = pd.read_csv(table_dir / f"{dataset}_complete.csv")
    stem = f"{dataset}_MAR_30per"
    mask = np.load(mask_dir / dataset / f"{stem}_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    missing = complete[feats].mask(mask_df)

    determinism.apply("deterministic", seed=seed)
    if method == "SNI":
        # SNI carries the SAME early-stopping configuration as TabCSDI --
        # patience 20, min_epochs 50, CosineAnnealingLR(T_max=epochs)
        # (sni/imputer.py:129-132, 827, 946). If that rule stops TabCSDI
        # prematurely it may stop SNI prematurely too, and relaxing it for the
        # competitor alone would be exactly the asymmetry R2-6 is about. So it
        # is measured on both.
        cfg = SNIConfig(seed=seed, use_gpu=True, epochs=epochs)
        if disable_early_stop:
            cfg.early_stopping_patience = epochs + 1
        imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                         continuous_vars=list(schema.continuous_vars), config=cfg)
        from sni.metrics import evaluate_imputation as _ev
        with _EpochCounter() as c:
            X_imp = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
        res = _ev(X_imputed=X_imp, X_complete=complete[feats], X_missing=missing,
                  categorical_vars=list(schema.categorical_vars),
                  continuous_vars=list(schema.continuous_vars), mask_df=mask_df)
        s_ = dict(res.summary) if hasattr(res, "summary") else dict(res)
        # SNI trains one model PER FEATURE inside an EM outer loop, so the
        # scheduler-step counter sums over all of them: 4233 steps on NHANES is
        # 3 EM iterations x 15 features x ~94 epochs each, not a 4233-epoch run.
        # The comparable quantity is epochs per per-feature model.
        n_models = max(1, len(feats) * int(getattr(cfg, "max_iters", 1)))
        per_model = c.n / n_models
        return {"method": "SNI", "dataset": dataset, "seed": seed,
                "epochs_cap": epochs, "epochs_run": c.n,
                "n_models_trained": n_models,
                "epochs_per_model": round(per_model, 1),
                "cap_was_binding": per_model >= epochs * 0.99,
                "early_stop_disabled": disable_early_stop,
                "cont_R2": s_.get("cont_R2"), "cont_NRMSE": s_.get("cont_NRMSE")}

    imp = build_baseline_imputer("TabCSDI",
                                 categorical_vars=list(schema.categorical_vars),
                                 continuous_vars=list(schema.continuous_vars),
                                 seed=seed, use_gpu=True, epochs=epochs)
    if disable_early_stop:
        # `TabCSDIBaseline` does not expose patience (registry.py:692-701) and
        # that dataclass is a faithful port, so it is not changed. The probe
        # reaches into the constructed implementation instead: a patience larger
        # than the cap can never fire, which makes the cap the binding constraint
        # so that "more budget" means what it says.
        imp._impl.early_stopping_patience = epochs + 1

    from sni.metrics import evaluate_imputation
    with _EpochCounter() as c:
        X_imp = imp.impute(missing, schema)
    res = evaluate_imputation(X_imputed=X_imp, X_complete=complete[feats],
                              X_missing=missing,
                              categorical_vars=list(schema.categorical_vars),
                              continuous_vars=list(schema.continuous_vars),
                              mask_df=mask_df)
    s = dict(res.summary) if hasattr(res, "summary") else dict(res)
    return {"method": "TabCSDI", "dataset": dataset, "seed": seed, "epochs_cap": epochs,
            "epochs_run": c.n, "cap_was_binding": c.n >= epochs,
            "early_stop_disabled": disable_early_stop,
            "cont_R2": s.get("cont_R2"), "cont_NRMSE": s.get("cont_NRMSE")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["NHANES", "Concrete"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--caps", type=int, nargs="*", default=[200, 800])
    ap.add_argument("--also-disable-early-stop", action="store_true")
    ap.add_argument("--methods", nargs="*", default=["TabCSDI"])
    ap.add_argument("--tables", default=str(CODE_ROOT / "data" / "derived_shuffled"))
    ap.add_argument("--masks", default=str(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"))
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before start (B48).",
              file=sys.stderr)
        return 2

    rows = []
    for ds in a.datasets:
        for cap in a.caps:
            for seed in a.seeds:
                for off in ([False, True] if a.also_disable_early_stop else [False]):
                  for meth in a.methods:
                    r = probe(ds, seed, cap, Path(a.tables), Path(a.masks), off, meth)
                    rows.append(r)
                    print(f"[{meth} {ds} cap={cap:<4} s{seed} "
                          f"{'no-earlystop' if off else 'default':<13}] "
                          f"epochs={r.get('epochs_per_model', r['epochs_run'])!s:<7} "
                          f"binding={str(r['cap_was_binding']):<5} "
                          f"R2={r['cont_R2']:>10.4f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    # Name the file after the methods it holds. The first version wrote a fixed
    # `epoch_probe.csv`, so running the probe for SNI silently overwrote the 16
    # TabCSDI rows; only the log survived to rebuild them from.
    tag = "_".join(sorted(set(df.method))) if "method" in df else "probe"
    df.to_csv(OUT / f"epoch_probe_{tag}.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUT / 'epoch_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
