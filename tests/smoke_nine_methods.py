"""T2b.3 / P2 section 4.4 criterion 6: do all nine methods run on all seven tables?

The last of the six criteria behind the MIMIC fallback gate, and the last check
before 55 GPU-hours of grid. One seed, MAR at 30 %, every method on every shipped
table. Success is not just "no exception": the metrics have to be of a sensible
magnitude, because a method that returns without error and an R^2 of -224 has
failed in the way that matters (finding B70).

Split by queue so it obeys the T2.0(c) policy -- GPU jobs never run two at a
time, CPU jobs fill the rest of the machine:

    python tests/smoke_nine_methods.py --queue cpu     # 7 methods
    python tests/smoke_nine_methods.py --queue gpu     # SNI, TabCSDI

Every failure is listed individually and none is skipped, which is the R0 lesson
(300 `baselines_deep` runs excluded by a comment in an aggregation script).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from baselines.registry import build_baseline_imputer      # noqa: E402
from sni.imputer import SNIConfig, SNIImputer               # noqa: E402
from baselines.schema import DataSchema                    # noqa: E402
from common import determinism                             # noqa: E402
from common.rowspace import RowSpace, assert_same_rowspace  # noqa: E402
from sni.metrics import evaluate_imputation                # noqa: E402

OUT = CODE_ROOT / "results" / "T2b_smoke"

#: What this check is for, and what it is NOT for.
#:
#: Criterion 6 asks whether the harness runs every method correctly. It does not
#: ask whether every method is any good, and it cannot ask whether our numbers
#: match R0's. Getting to that took three passes, each of which found the
#: criterion measuring the wrong thing:
#:
#: 1. An absolute floor of R^2 > -1 ("R^2 above the column-mean baseline") flagged GAIN on
#:    NHANES at -2.635. R0's own record has GAIN negative on all six datasets
#:    (median -0.27 to -4.15), so that was a faithfully reproduced weak baseline
#:    being called a harness failure.
#: 2. Comparing against R0's per-(method, dataset) range instead. But the "R0
#:    still applies" set wrongly included eICU (20 features -> 16) and Concrete
#:    (Duration retyped, categorical count 1 -> 0). GAIN/eICU was then flagged
#:    for scoring *above* R0's band on a table R0 never had.
#: 3. Restricting the comparison to the two unchanged tables. That failed too:
#:    GAIN/AutoMPG came out at +0.028 against R0's [-1.562, -0.839] -- because
#:    **the masks changed on every dataset**. R0's MAR was driven by the row
#:    index; ours is driven by model_year with per-column coefficients. Same
#:    table, different problem. R0's numbers are not a reference anywhere.
#:
#: So the pass/fail rule is now only what can be defended: the run completed, the
#: metrics are finite, and they are not absurd. The R0 delta is still computed
#: and reported, as **context** -- a large gap is worth looking at -- but it does
#: not decide the verdict, because nothing about it is held constant.
#:
#: R^2 below -10 or NRMSE above 2 is a diverged fit, not a weak one: TabCSDI
#: reaches -224 on NHANES (B70) while GAIN's genuine worst is about -8.9.
R2_ABSOLUTE_FLOOR = -10.0
NRMSE_CEIL = 2.0
#: Tables whose contents changed in R1. Kept for the descriptive R0 delta, which
#: is only meaningful where the table itself is unchanged.
REBUILT = {"MIMIC", "NHANES", "CDC2022", "eICU", "Concrete"}



def r0_reference() -> dict:
    """R0's own recorded R^2 range per (method, dataset), pooled over its
    aggregation directories. Used as the plausibility reference for the tables we
    did not change; returns {} if the R0 tree is absent."""
    import glob
    root = CODE_ROOT.parent / "project_sni_R0" / "results_all"
    files = glob.glob(str(root / "agg_baselines_*" / "summary_all.csv"))
    if not files:
        return {}
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d.mechanism == "MAR") & (d.rate == "30per")]
    out = {}
    for (algo, ds), g in d.groupby(["algo", "dataset"]):
        r2 = g.cont_R2.dropna()
        if len(r2):
            out[(algo, ds)] = (float(r2.min()), float(r2.max()))
    return out


def plausibility(method: str, dataset: str, r2, nrmse, ref: dict) -> list:
    """Reasons this run looks like a pipeline problem rather than a weak method.

    Absolute sanity only -- see the note above `R2_ABSOLUTE_FLOOR` for why the
    R0 comparison was demoted to context.
    """
    problems = []
    if r2 is not None and not np.isfinite(r2):
        problems.append("non-finite R2")
    if nrmse is not None and not np.isfinite(nrmse):
        problems.append("non-finite NRMSE")
    if nrmse is not None and np.isfinite(nrmse) and nrmse > NRMSE_CEIL:
        problems.append(f"NRMSE={nrmse:.3f} > {NRMSE_CEIL}")
    if r2 is not None and np.isfinite(r2) and r2 < R2_ABSOLUTE_FLOOR:
        problems.append(f"R2={r2:.3f} < {R2_ABSOLUTE_FLOOR} (diverged, not weak)")
    return problems


def r0_context(method: str, dataset: str, r2, ref: dict) -> str:
    """Descriptive only. Blank where R0 has no comparable record."""
    if dataset in REBUILT or r2 is None or not np.isfinite(r2):
        return ""
    band = ref.get((method, dataset))
    if not band:
        return ""
    lo, hi = band
    inside = lo <= r2 <= hi
    return (f"R0 range [{lo:.3f}, {hi:.3f}]"
            + ("" if inside else "; ours is outside it, expected because the "
                                 "mask mechanism changed"))


def load(dataset: str, table_dir: Path, mask_dir: Path):
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    blk = cfg["datasets"][dataset]
    complete = pd.read_csv(table_dir / f"{dataset}_complete.csv")

    stem = f"{dataset}_MAR_30per"
    meta = json.loads((mask_dir / dataset / f"{stem}_meta.json").read_text())
    mask = np.load(mask_dir / dataset / f"{stem}_mask.npy").astype(bool)

    # P2b decision 3: refuse to evaluate a table against a mask from another row
    # space. This is the assertion, at the place it matters.
    assert_same_rowspace(
        RowSpace.of_frame(complete, f"{dataset}:table",
                          blk.get("identifier_column", "ID"),
                          str(table_dir / f"{dataset}_complete.csv")),
        RowSpace.of_mask_meta(meta, f"{dataset}:mask", str(mask_dir)))

    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask, columns=list(complete.columns))[feats]
    missing = complete[feats].mask(mask_df)
    return complete, missing, mask_df, schema, feats


def run_one(method: str, dataset: str, seed: int, use_gpu: bool,
            table_dir: Path, mask_dir: Path, ref: dict = None) -> dict:
    complete, missing, mask_df, schema, feats = load(dataset, table_dir, mask_dir)
    determinism.apply("deterministic", seed=seed)
    # SNI is the method under study, not a registry baseline -- see
    # experiments/run_grid.py:_build for why the two paths differ.
    #
    # The P2c training protocol is applied here too, so these timings are the
    # timings the grid will actually see. Without it the smoke test would
    # measure the old early-stopping behavior and the schedule would have to
    # multiply an extrapolation by an estimated ratio -- two guesses stacked.
    from experiments.run_grid import _apply_training_protocol
    if method == "SNI":
        imp = SNIImputer(categorical_vars=list(schema.categorical_vars),
                         continuous_vars=list(schema.continuous_vars),
                         config=SNIConfig(seed=seed, use_gpu=use_gpu))
        proto = _apply_training_protocol(method, imp, True)
        t0 = time.time()
        # mask_df is passed explicitly: SNI evaluates on the declared mask rather
        # than re-deriving it from NaNs, matching tests/gate_equivalence.py:151.
        X_imp = imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
    else:
        imp = build_baseline_imputer(method,
                                     categorical_vars=list(schema.categorical_vars),
                                     continuous_vars=list(schema.continuous_vars),
                                     seed=seed, use_gpu=use_gpu)
        proto = _apply_training_protocol(method, imp, False)
        t0 = time.time()
        X_imp = imp.impute(missing, schema)
    elapsed = time.time() - t0

    res = evaluate_imputation(X_imputed=X_imp, X_complete=complete[feats],
                              X_missing=missing,
                              categorical_vars=list(schema.categorical_vars),
                              continuous_vars=list(schema.continuous_vars),
                              mask_df=mask_df)
    s = dict(res.summary) if hasattr(res, "summary") else dict(res)

    problems = plausibility(method, dataset, s.get("cont_R2"),
                            s.get("cont_NRMSE"), ref or {})
    context = r0_context(method, dataset, s.get("cont_R2"), ref or {})

    s.update(method=method, dataset=dataset, seed=seed, runtime_sec=elapsed,
             status="ok" if not problems else "IMPLAUSIBLE",
             problems="; ".join(problems), r0_context=context, **proto)
    return s


def main() -> int:
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    sched = yaml.safe_load((CODE_ROOT / "configs" / "scheduling.yaml").read_text())
    placement = sched["method_placement"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["cpu", "gpu", "all"], default="all")
    ap.add_argument("--datasets", nargs="*", default=list(cfg["datasets"]))
    # Needed to fill a single hole without re-running its neighbors. SNI on
    # CDC2022 was stopped after 7.2 h (37 imputable columns, 30 of them
    # categorical, so 111 per-feature fits at ~112 one-hot input dims); TabCSDI
    # on the same table still needs measuring, and without this flag the only
    # way to get it is to pay for the SNI cell again.
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these methods within the selected queue")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tables", default=str(CODE_ROOT / "data" / "derived_shuffled"))
    ap.add_argument("--masks", default=str(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"))
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts (finding B48).", file=sys.stderr)
        return 2

    methods = [m for m, q in placement.items()
               if a.queue == "all" or q == a.queue]
    if a.methods:
        unknown = set(a.methods) - set(methods)
        if unknown:
            print(f"REFUSING TO RUN: --methods names {sorted(unknown)}, which are "
                  f"not in queue '{a.queue}' ({methods})", file=sys.stderr)
            return 2
        methods = [m for m in methods if m in a.methods]
    OUT.mkdir(parents=True, exist_ok=True)

    ref = r0_reference()
    print(f"R0 reference bands available for {len(ref)} (method, dataset) pairs")

    rows, failures = [], []
    for ds in a.datasets:
        for m in methods:
            tag = f"{m}/{ds}"
            try:
                r = run_one(m, ds, a.seed, placement[m] == "gpu",
                            Path(a.tables), Path(a.masks), ref)
            except Exception as exc:
                failures.append({"run": tag, "error": repr(exc)[:400],
                                 "traceback": traceback.format_exc()[-1200:]})
                print(f"[FAIL] {tag}: {exc!r}", flush=True)
                continue
            rows.append(r)
            flag = "" if r["status"] == "ok" else f"  <== {r['problems']}"
            print(f"[{r['status']:<11}] {tag:<24} "
                  f"R2={r.get('cont_R2', float('nan')):>9.4f} "
                  f"NRMSE={r.get('cont_NRMSE', float('nan')):.5f} "
                  f"{r['runtime_sec']:6.1f}s{flag}", flush=True)

    df = pd.DataFrame(rows)
    # The output name must be unique per invocation, not per queue. The CPU arm
    # was run as three workers split by --datasets, and all three wrote
    # `smoke_cpu.csv`: the last to finish won and **42 of 49 rows were lost**.
    # They were recovered only because `tests/rescore_smoke.py` rebuilds from the
    # per-worker logs. Same class as the epoch-probe overwrite. Deriving the
    # suffix from the dataset selection makes concurrent workers collision-proof
    # without the launcher having to remember to pass anything.
    suffix = a.queue
    if set(a.datasets) != set(cfg["datasets"]):
        suffix += "_" + "-".join(sorted(a.datasets))[:60]
    if a.methods:
        suffix += "_" + "-".join(sorted(a.methods))[:30]
    df.to_csv(OUT / f"smoke_{suffix}.csv", index=False)
    (OUT / f"failures_{suffix}.json").write_text(json.dumps(failures, indent=2))
    print(f"wrote {OUT}/smoke_{suffix}.csv ({len(df)} rows)")

    n_bad = int((df.status != "ok").sum()) if not df.empty else 0
    print("\n" + "=" * 70)
    print(f"{len(df)} run(s) completed, {len(failures)} crashed, "
          f"{n_bad} implausible")
    if failures:
        for f in failures:
            print(f"  CRASH  {f['run']}: {f['error'][:120]}")
    if n_bad:
        for _, r in df[df.status != "ok"].iterrows():
            print(f"  IMPLAUSIBLE  {r.method}/{r.dataset}: {r.problems}")
    return 0 if not failures and not n_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
