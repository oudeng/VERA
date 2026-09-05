"""T2.3: the full re-run grid.

9 methods x 7 datasets x 3 mechanisms x 3 rates x 5 seeds = 2835 runs, plus the
45-run real-pattern condition from T2b.2, for 2880 total.

What this one grid settles, per the P2 instruction: R1-4 (the new MAR),
R2-6b (MNAR for HyperImpute and TabCSDI, which were never in the plan at all),
B3 (the undisclosed multi-rate coverage gap -- the most dangerous defect the
reviewers did *not* find), B6 (oracle leakage), B34/B35/B59 (data layer),
B48/B66 (seed reproducibility) and B49 (cross-rate mechanism strength).

Non-negotiables, each traceable to something that went wrong in R0:

* **No silent skips.** R0 lost 300 `baselines_deep` runs to a comment in an
  aggregation script and the paper never mentioned it. Every failure here is
  written to `error.log`, counted, and listed individually in the report.
* **PYTHONHASHSEED before the interpreter starts** (B48), asserted at entry.
* **Row-space assertion** at every (table, mask) pairing (P2b decision 3): the
  shuffled masks index shuffled tables, and the wrong pairing raises nothing.
* **Masks load from `.npy` and are asserted against the table** (E4, B38).
* **One `run_config.json` per run** (E2).
* **GPU jobs never run two at a time** (T2.0(c): measured 13-15x penalty).
* **Median is the primary aggregate**, applied to every method and metric alike
  (P2b decision 1), with the divergence rate recorded for all of them.

Resume is by artifact: a run whose `metrics_summary.json` exists is skipped
unless `--no-skip-existing`. That makes the grid restartable after a crash
without re-doing GPU-hours.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# THREAD COUNT IS A CONTROLLED VARIABLE (finding B84). Must precede numpy/torch.
#
# Measured on MIMIC seed 1, port side, identical seed / hash_probe / EM
# iterations, BLAS thread count the only difference:
#
#     cat_Macro-F1   spread 5.20e-2      cat_Accuracy  3.75e-2
#     cat_Cohen_kappa      3.83e-2      cont_R2       3.34e-3
#
# For scale, R0's Table S3 headline is SNI beating GAIN by Macro-F1 +0.236, so
# thread count alone moves that metric by 22 % of the effect the paper reports.
# Until this was pinned, every grid cell had an uncontrolled variable in it.
#
# Default 2: the CPU queue runs 12 wide (T2c.1), and 12 x 2 = 24 threads leaves
# headroom on 32 cores for the GPU worker's host thread and for I/O. Override
# with SNI_NUM_THREADS -- T2f.2 needs to vary it deliberately.
# ---------------------------------------------------------------------------
_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


def _thread_provenance() -> dict:
    """What the run actually got, not what it asked for.

    B48's lesson applied to a second environment variable: an env var set from
    Python is a request, not a fact. `torch.get_num_threads()` is the fact.
    """
    import torch
    want = int(_NT)
    got = int(torch.get_num_threads())
    blas = ""
    try:
        blas = torch.__config__.show()
    except Exception:
        pass
    return {"threads_requested": want, "threads_actual": got,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "torch_version": torch.__version__,
            "torch_config": blas[:2000],
            "numpy_version": np.__version__}

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from baselines.registry import build_baseline_imputer          # noqa: E402
from sni.imputer import SNIConfig, SNIImputer                   # noqa: E402
from baselines.schema import DataSchema                        # noqa: E402
from common import determinism, runconfig                      # noqa: E402
from common import masks as common_masks                       # noqa: E402
from common.rowspace import RowSpace, assert_same_rowspace     # noqa: E402
from sni.metrics import evaluate_imputation                    # noqa: E402

OUT = CODE_ROOT / "results" / "P2_main_grid"
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.3, 0.5]
SEEDS = [1, 2, 3, 5, 8]


# ---------------------------------------------------------------------------

def _cfg(name: str) -> dict:
    return yaml.safe_load((CODE_ROOT / "configs" / f"{name}.yaml").read_text())


def _rate_tag(rate: float) -> str:
    return f"{int(round(float(rate) * 100)):02d}per"


def load_case(dataset: str, mechanism: str, rate: Optional[float], seed: int,
              table_dir: Path, mask_dir: Path):
    """Table, mask and schema for one cell, with both assertions applied."""
    blk = _cfg("datasets")["datasets"][dataset]
    ident = blk.get("identifier_column", "ID")
    tpath = table_dir / f"{dataset}_complete.csv"
    complete = pd.read_csv(tpath)

    if mechanism == "REAL_PATTERN":
        stem = f"{dataset}_REALPATTERN_s{seed}"
        mroot = CODE_ROOT / "data" / "masks" / "real_pattern"
    else:
        stem = f"{dataset}_{mechanism}_{_rate_tag(rate)}"
        mroot = mask_dir
    mpath = mroot / dataset / f"{stem}_mask.npy"
    meta = json.loads((mroot / dataset / f"{stem}_meta.json").read_text())

    # P2b decision 3. The one pairing error that produces no error.
    assert_same_rowspace(
        RowSpace.of_frame(complete, f"{dataset}:table", ident, str(tpath)),
        RowSpace.of_mask_meta(meta, f"{dataset}:{stem}", str(mpath)))

    mask_full = np.load(mpath).astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    # P5R SS0.2 assertion (internal review SS4.8): the downstream target is
    # never an imputer feature, and although the table-shaped mask files may
    # carry a mask column for it (the executed MNAR configs do), that column
    # is dropped here with everything outside the feature schema before any
    # imputer or metric sees the data. Asserted, not assumed. The review's
    # literal formula `target not in maskable` is unsatisfiable at the
    # mask-file level for the executed grid; the protocol-true invariant is
    # asserted instead and the deviation is recorded in the P5R receipt.
    import yaml as _yaml
    _tgt = _yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml"
                            ).read_text())["datasets"][dataset].get(
        "downstream_target")
    assert _tgt not in feats, \
        f"{dataset}: downstream target {_tgt!r} is in the imputer schema"
    mask_df = pd.DataFrame(mask_full, columns=list(complete.columns))[feats]
    assert _tgt not in mask_df.columns, \
        f"{dataset}: target mask column survived the feature restriction"
    print(f"target-exclusion assertion passed: {_tgt!r} not in features, "
          f"mask restricted to {len(feats)} feature columns", flush=True)
    missing = complete[feats].mask(mask_df)

    # E4: the cached .npy is the authority, asserted against the table it masks.
    _, chk = common_masks.load_and_verify(
        complete.mask(pd.DataFrame(mask_full, columns=list(complete.columns))),
        mpath, columns=list(complete.columns), strict=True)
    if not chk.consistent:
        raise RuntimeError(f"E4 mask/table disagreement for {stem}")

    # Content hash of the mask file itself.
    #
    # The row-space digest catches a mask built against a *different row order*.
    # It cannot catch a mask whose content changed while the row order stayed the
    # same -- which is exactly what happened during P2b when decision 2(b)
    # regenerated NHANES's masks 7 minutes into a running experiment: same
    # shuffled table, same row order, different driver set, and nothing raised.
    # Half that experiment's arms used a different input from the other half and
    # the only reason it surfaced was a follow-up run disagreeing with it.
    #
    # Recorded per run so aggregation can assert that every seed of a cell saw
    # the same mask. A 2.3-day grid gives plenty of opportunity to repeat the
    # mistake.
    mask_md5 = hashlib.md5(mpath.read_bytes()).hexdigest()[:16]
    return complete, missing, mask_df, schema, feats, mask_md5



def _build(method: str, schema, seed: int, use_gpu: bool):
    """SNI is not in the baseline registry.

    `baselines/registry.py` holds the eight *competitors*; SNI is the method
    under study and lives in `sni/imputer.py` with its own config object. The
    first version of this file called `build_baseline_imputer("SNI", ...)`, which
    would have raised on all 320 SNI cells -- caught here rather than 55
    GPU-hours in, though the T2b.3 smoke test would also have caught it.

    SNI additionally produces artifacts nothing else does, and the later tasks
    need them: the dependency matrix D (T2.5's R2-1 pilot and R2-4's stability
    curve) and the per-head lambda record (R2-5). They are written per run,
    because regenerating them means re-running the imputation.
    """
    if method == "SNI":
        return SNIImputer(categorical_vars=list(schema.categorical_vars),
                          continuous_vars=list(schema.continuous_vars),
                          config=SNIConfig(seed=seed, use_gpu=use_gpu)), True
    return build_baseline_imputer(
        method, categorical_vars=list(schema.categorical_vars),
        continuous_vars=list(schema.continuous_vars),
        seed=seed, use_gpu=use_gpu), False



def _apply_training_protocol(method: str, imp, is_sni: bool) -> dict:
    """P2c option (d): run the configured budget; do not stop early.

    Reads `configs/training_protocol.yaml` rather than hard-coding, so the one
    place that says what the protocol is stays the one place that decides it (E1).

    Neither dataclass is edited. `TabCSDIBaseline` and `SNIConfig` are faithful
    ports and the equivalence gate rests on them; the patience is raised on the
    constructed object instead, which is a run-time choice and is recorded in
    every run_config.json.

    Only SNI and TabCSDI have such a rule -- GAIN runs a fixed
    `iterations=10000` loop and MIWAE's loop is annotated "no early stopping"
    (audit table in configs/training_protocol.yaml).
    """
    proto = _cfg("training_protocol")["protocol"]
    if not proto.get("disable_early_stopping"):
        return {"training_protocol": "as_configured_with_early_stopping"}

    epochs = proto["epochs"].get(method)
    if epochs is None:
        return {"training_protocol": "unaffected", "early_stopping_disabled": False}

    # patience > epochs can never fire, so the configured budget is what runs.
    if is_sni:
        imp.cfg.epochs = int(epochs)
        imp.cfg.early_stopping_patience = int(epochs) + 1
    else:
        imp._impl.epochs = int(epochs)
        imp._impl.early_stopping_patience = int(epochs) + 1
    return {"training_protocol": proto["name"], "epochs_configured": int(epochs),
            "early_stopping_disabled": True}


def run_cell(method: str, dataset: str, mechanism: str, rate: Optional[float],
             seed: int, use_gpu: bool, table_dir: Path, mask_dir: Path,
             outdir: Path) -> dict:
    complete, missing, mask_df, schema, feats, mask_md5 = load_case(
        dataset, mechanism, rate, seed, table_dir, mask_dir)

    det_state = determinism.apply("deterministic", seed=seed)
    imp, is_sni = _build(method, schema, seed, use_gpu)
    proto_rec = _apply_training_protocol(method, imp, is_sni)
    t0 = time.time()
    X_imp = (imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
             if is_sni else imp.impute(missing, schema))
    elapsed = time.time() - t0

    res = evaluate_imputation(X_imputed=X_imp, X_complete=complete[feats],
                              X_missing=missing,
                              categorical_vars=list(schema.categorical_vars),
                              continuous_vars=list(schema.continuous_vars),
                              mask_df=mask_df)
    s = dict(res.summary) if hasattr(res, "summary") else dict(res)
    s.update(method=method, dataset=dataset, mechanism=mechanism,
             rate=(float(rate) if rate is not None else None), seed=seed,
             use_gpu=use_gpu, runtime_sec=elapsed, mask_md5=mask_md5,
             **proto_rec)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics_summary.json").write_text(json.dumps(s, indent=2, default=str))

    # SNI-only artifacts. D is what reviewer point R2-1 is about and what R2-4
    # measures the stability of; the per-head lambda record is R2-5's evidence.
    # Both are free here and expensive to recover later.
    if is_sni:
        try:
            imp.compute_dependency_matrix().to_csv(outdir / "dependency_matrix.csv")
            imp.get_lambda_per_head_df().to_csv(outdir / "lambda_per_head.csv",
                                                index=False)
            (outdir / "lambda_summary.json").write_text(
                json.dumps(imp.get_lambda_summary(), indent=2, default=str))
        except Exception as exc:                # recorded, never swallowed
            (outdir / "artifact_error.txt").write_text(repr(exc))
    X_imp.to_csv(outdir / "imputed.csv", index=False, float_format="%.17g")
    # E2: the config that actually ran, not the one that was requested --
    # `determinism.apply` returns the policy it managed to install.
    runconfig.write(outdir, runconfig.build(
        exp_id=outdir.name,
        method=method,
        params={"use_gpu": use_gpu, "categorical_vars": list(schema.categorical_vars),
                "continuous_vars": list(schema.continuous_vars), **proto_rec,
                "thread_provenance": _thread_provenance()},
        inputs={"dataset": dataset, "mechanism": mechanism, "rate": rate,
                "table_dir": str(table_dir), "mask_dir": str(mask_dir),
                "mask_md5": mask_md5,
                "n_rows": int(len(complete)), "n_features": len(feats)},
        seeds={"run": seed,
               # The startup value, not os.environ -- seed_everything
               # overwrites the variable with the per-run seed (B48).
               "pythonhashseed": determinism.STARTUP_PYTHONHASHSEED},
        determinism=det_state,
        extra={"task": "P2_main_grid", "runtime_sec": elapsed}))
    return s


#: P2c section 4: CDC2022 does not need the full 3x3.
#:
#: It serves two purposes -- R2-4's "real tables are wider" D-stability curve,
#: and the real-pattern condition -- and neither needs three mechanisms at three
#: rates. Cutting it to MCAR+MAR at 30 % drops 315 cells.
#:
#: The saving is concentrated exactly where it hurts: KNN is a Python double
#: loop over donor rows and measures ~1.5 h per run on this table, so its share
#: alone falls from 45 cells (67 h) to 10 (15 h) and the CPU queue drops back
#: below the GPU queue. The other six tables keep the full 3x3, because filling
#: B3's undisclosed multi-rate gap is what the grid is for.
SLIM = {
    "CDC2022": {"mechanisms": ["MCAR", "MAR"], "rates": [0.3]},
}


def cells(datasets: Iterable[str], methods: Iterable[str],
          include_real_pattern: bool) -> List[tuple]:
    out = []
    for d in datasets:
        mechs = SLIM.get(d, {}).get("mechanisms", MECHANISMS)
        rates = SLIM.get(d, {}).get("rates", RATES)
        out += [(m, d, mech, r, s)
                for m in methods for mech in mechs for r in rates for s in SEEDS]
    if include_real_pattern:
        out += [(m, "CDC2022", "REAL_PATTERN", None, s)
                for m in methods for s in SEEDS]
    return out


def main() -> int:
    ds_cfg = _cfg("datasets")["datasets"]
    placement = _cfg("scheduling")["method_placement"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["cpu", "gpu", "all"], default="all")
    ap.add_argument("--datasets", nargs="*", default=list(ds_cfg))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tables", default=str(CODE_ROOT / "data" / "derived_shuffled"))
    ap.add_argument("--masks", default=str(CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shard", default=None,
                    help="i/n -- take every n-th cell starting at i. Partitions "
                         "by CELL, not by dataset: KNN is a Python double loop "
                         "over donor rows (baselines/KNN_v1.py:255-262), so on "
                         "CDC2022 it is ~1.5 h per run and its 45 cells alone "
                         "would pin one dataset-sharded worker for 67 h -- "
                         "longer than the entire GPU queue.")
    # Needed by T2f.3's bit-identical re-verification, which must repeat ONE cell
    # rather than a 45-cell dataset sweep. Sharding cannot express "just this
    # mechanism, rate and seed".
    ap.add_argument("--mechanisms", nargs="*", default=None)
    ap.add_argument("--rates", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--no-real-pattern", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the interpreter "
              "starts. Setting it from inside has no effect -- that is B48, and "
              "it is why no R0 SNI or MICE result is reproducible from its seed.",
              file=sys.stderr)
        return 2

    # B84: the thread count is a controlled variable now, and an env var set from
    # Python is a request rather than a fact (B48's lesson). Check it.
    _prov = _thread_provenance()
    if _prov["threads_actual"] != _prov["threads_requested"]:
        print(f"REFUSING TO RUN: asked for {_prov['threads_requested']} BLAS "
              f"threads but torch reports {_prov['threads_actual']}. Thread count "
              f"changes SNI's metrics (B84: Macro-F1 by 5.2e-2), so an "
              f"unverified setting makes every cell uncontrolled.",
              file=sys.stderr)
        return 2
    print(f"threads pinned at {_prov['threads_actual']} (verified); "
          f"torch {_prov['torch_version']}, numpy {_prov['numpy_version']}")

    methods = a.methods or [m for m, q in placement.items()
                            if a.queue == "all" or q == a.queue]

    # A CPU-queue worker must not open a CUDA context. HyperImpute does, without
    # anything in its code saying so -- about 440 MiB per process. On its own
    # that is harmless; running twelve of them beside the GPU queue is not.
    # Measured (T2d.2): TabCSDI alone 328.9 s; beside 12 HyperImpute workers
    # 958.9 s (**2.92x**); beside the same 12 with CUDA hidden 342.5 s (1.04x).
    # The schedule assumes the two queues cost each other about 1 %, and that
    # assumption was only ever checked with one CPU job beside one GPU job
    # (gate T2.0(c), n=10). Left alone this would have turned a 238 h GPU queue
    # into roughly 700 h.
    #
    # Hiding the device is the cleanest of the three remedies P2d lists: no queue
    # is re-assigned and no concurrency is throttled. It is set here rather than
    # in the launcher so it cannot be forgotten by whoever writes the next chain
    # script -- correctness should not depend on remembering an env var.
    if a.queue == "cpu" and not a.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Setting an environment variable inside a running interpreter is exactly
        # the shape of B48, where `PYTHONHASHSEED` was set from Python and had no
        # effect at all. This one *does* work -- CUDA reads it when the context is
        # first created, not at interpreter start -- but only while nothing has
        # initialized CUDA during imports. That is a precondition, not a
        # guarantee, so it is checked rather than assumed: a silent failure here
        # costs 2.92x on the whole GPU queue and would look like nothing.
        import torch
        if torch.cuda.device_count() != 0:
            print("REFUSING TO RUN: CUDA is still visible to this CPU-queue "
                  "worker after CUDA_VISIBLE_DEVICES was set empty, which means "
                  "something initialized CUDA during import. Launch with "
                  "CUDA_VISIBLE_DEVICES= set in the environment instead (T2d.2, "
                  "finding B82).", file=sys.stderr)
            return 2
        print("CPU queue: CUDA hidden, verified device_count=0 (T2d.2, B82)")
    todo = cells(a.datasets, methods, not a.no_real_pattern)
    if a.mechanisms:
        todo = [c for c in todo if c[2] in a.mechanisms]
    if a.rates is not None:
        keep = {None} | {float(r) for r in a.rates}
        todo = [c for c in todo if c[3] in keep]
    if a.seeds:
        todo = [c for c in todo if c[4] in set(a.seeds)]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        todo = [c for k, c in enumerate(todo) if k % n == i]
        print(f"shard {i}/{n}: {len(todo)} cells")
    root = Path(a.out)

    if a.dry_run:
        print(f"{len(todo)} cells; methods={methods}; datasets={a.datasets}")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    errlog = root / "error.log"
    done, skipped, failed = [], 0, []

    for i, (m, d, mech, r, s) in enumerate(todo, 1):
        tag = (f"{d}_{mech}_{_rate_tag(r)}_{m}_s{s}" if r is not None
               else f"{d}_{mech}_{m}_s{s}")
        outdir = root / tag
        if (outdir / "metrics_summary.json").exists() and not a.no_skip_existing:
            skipped += 1
            continue
        try:
            res = run_cell(m, d, mech, r, s, placement[m] == "gpu",
                           Path(a.tables), Path(a.masks), outdir)
        except Exception as exc:
            rec = {"run": tag, "error": repr(exc)[:400],
                   "traceback": traceback.format_exc()[-1500:]}
            failed.append(rec)
            with errlog.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[FAIL {i}/{len(todo)}] {tag}: {exc!r}", flush=True)
            continue
        done.append(res)
        print(f"[OK {i}/{len(todo)}] {tag:<48} "
              f"R2={res.get('cont_R2', float('nan')):>9.4f} "
              f"{res['runtime_sec']:7.1f}s", flush=True)

    print("\n" + "=" * 72)
    print(f"completed {len(done)}, skipped {skipped} (already present), "
          f"FAILED {len(failed)}")
    if failed:
        print(f"failures written to {errlog}, listed individually:")
        for f in failed:
            print(f"  {f['run']}: {f['error'][:140]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
