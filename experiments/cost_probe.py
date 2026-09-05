"""A3 cost contextualisation probe (P5R-B SS4-A3; internal review SS6.4).

Single-threaded, standardized timing + peak memory for one representative
cell per real table (the faithfulness condition: MAR@30 mask, seed 1), one
audit object per process, so the published 47-64x grid-sourced ratio gets
the review-requested sensitivity: a ratio measured with no concurrency, no
queue contention, pinned to one BLAS thread, with peak RSS alongside.
DESCRIPTIVE ONLY -- no verdict keys on these numbers; the paper sentence
they support is "runtime is a record of the current implementation under
the stated hardware", plus the single-thread ratio.

Protocol (fixed):
  * SNI_NUM_THREADS=1 enforced (the module refuses otherwise);
  * idle-machine guard: refuses to start if 1-min load average > 2.0;
  * one object per invocation: P | MF | SNI; peak RSS is read by the
    caller via /usr/bin/time -v (driver: scripts snippet in the receipt);
  * walls recorded per phase (impute/train, readout) with
    time.monotonic(); environment block records CPU model, RAM, commit,
    library versions.

    env PYTHONHASHSEED=2025 SNI_NUM_THREADS=1 \
      /usr/bin/time -v python experiments/cost_probe.py --dataset MIMIC \
      --object SNI 2> results/A3_cost_context/MIMIC_SNI_time.txt
"""
from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = CODE_ROOT / "results" / "A3_cost_context"
SEED = 1
#: P5R-H SS3 (third review SS7.3): each MissForest readout probed in its own
#: process so peak RSS is per audit object, not a shared host-family envelope.
#: The shared forest fit is included in every one of them, which is the honest
#: accounting: no readout exists without it.
MF_READOUTS = {"MFimp": "MissForest-importance",
               "MFshap": "SHAP-on-MissForest",
               "MFperm": "Permutation-on-MissForest"}


def _env_block() -> dict:
    from common import runconfig
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            mem_kb = int(line.split()[1])
            break
    import subprocess
    topo = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    topo_lines = [ln.strip() for ln in topo.splitlines()
                  if ln.split(":")[0].strip() in
                  ("CPU(s)", "Thread(s) per core", "Core(s) per socket",
                   "Model name")]
    return {"cpu_model": cpu, "mem_total_gb": round(mem_kb / 1e6, 1),
            "nproc_online": os.cpu_count(),
            "threads_pinned": 1,
            "cpu_topology": topo_lines,
            "cpu_note": ("heterogeneous P/E cores: a lone single-thread "
                         "process is scheduled on a performance core at "
                         "full boost, so these walls are the P-core best "
                         "case (observed contended-vs-lone ratio up to "
                         "~3.7x on this machine); cross-machine "
                         "comparability is part of the SS6.4 "
                         "contextualisation text"),
            "code_SNI_commit": runconfig.git_commit(),
            "libraries": runconfig.library_versions()}


def probe(ds: str, obj: str) -> dict:
    from prior_attribution import compute_P, load_real_case
    missing, mask_df, cat, cont = load_real_case(ds)
    rec: dict = {"dataset": ds, "object": obj, "seed": SEED,
                 "mask": "MAR_30per", "n_rows": int(len(missing)),
                 "n_features": len(cat) + len(cont)}
    if obj == "P":
        t0 = time.monotonic()
        compute_P(cat, cont, SEED, missing, mask_df)
        rec["wall_total_sec"] = round(time.monotonic() - t0, 3)
    elif obj in ("MF",) + tuple(MF_READOUTS):
        from pilot_r21 import run_missforest_family
        sel = None if obj == "MF" else [MF_READOUTS[obj]]
        t0 = time.monotonic()
        fam, _no_model = run_missforest_family(
            missing, cat, cont, SEED, cat + cont, readouts=sel)
        rec["wall_total_sec"] = round(time.monotonic() - t0, 3)
        rec["readouts_computed"] = sorted(fam)
        rec["per_readout_sec"] = {name: {"impute_sec": round(s1, 3),
                                         "readout_sec": round(s2, 3)}
                                  for name, (_M, s1, s2) in fam.items()}
    elif obj == "SNI":
        import yaml
        from common import determinism
        from sni.imputer import SNIConfig, SNIImputer
        proto = yaml.safe_load((CODE_ROOT / "configs"
                                / "training_protocol.yaml").read_text()
                               )["protocol"]
        determinism.apply("deterministic", seed=SEED)
        imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                         config=SNIConfig(seed=SEED, use_gpu=False))
        imp.cfg.epochs = int(proto["epochs"]["SNI"])
        imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
        t0 = time.monotonic()
        imp.impute(X_missing=missing, X_complete=None, mask_df=mask_df)
        t1 = time.monotonic()
        imp.compute_dependency_matrix()
        t2 = time.monotonic()
        rec.update({"wall_train_impute_sec": round(t1 - t0, 3),
                    "wall_D_readout_sec": round(t2 - t1, 3),
                    "wall_total_sec": round(t2 - t0, 3)})
    elif obj == "PermSNI":
        # Same-host behavioral probe: the SNI host fit (shared with the
        # SNI-D object) plus the permutation-ablation readout on that host.
        import yaml
        from common import determinism
        from sni.imputer import SNIConfig, SNIImputer
        proto = yaml.safe_load((CODE_ROOT / "configs"
                                / "training_protocol.yaml").read_text()
                               )["protocol"]
        determinism.apply("deterministic", seed=SEED)
        imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                         config=SNIConfig(seed=SEED, use_gpu=False))
        imp.cfg.epochs = int(proto["epochs"]["SNI"])
        imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
        t0 = time.monotonic()
        completed = imp.impute(X_missing=missing, X_complete=None,
                               mask_df=mask_df)
        t1 = time.monotonic()
        from t4f_perm_on_sni import _ablate
        complete = pd.read_csv(CODE_ROOT / "data" / "derived_shuffled"
                               / f"{ds}_complete.csv")[cat + cont]
        _ablate(imp, completed, mask_df, complete, imp.all_vars, SEED,
                set(cat))
        t2 = time.monotonic()
        rec.update({"wall_train_impute_sec": round(t1 - t0, 3),
                    "wall_readout_sec": round(t2 - t1, 3),
                    "wall_total_sec": round(t2 - t0, 3)})
    else:
        raise ValueError(obj)
    rec["environment"] = _env_block()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MIMIC", "eICU"])
    ap.add_argument("--object", required=True,
                    choices=["P", "MF", "SNI", "PermSNI"] + list(MF_READOUTS))
    ap.add_argument("--force-load", action="store_true",
                    help="skip the idle-machine guard (never for real runs)")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set (finding B48).",
              file=sys.stderr)
        return 2
    if _NT != "1":
        print("REFUSING TO RUN: SNI_NUM_THREADS=1 is the protocol "
              f"(got {_NT!r}).", file=sys.stderr)
        return 2
    load1 = os.getloadavg()[0]
    if load1 > 2.0 and not a.force_load:
        print(f"REFUSING TO RUN: 1-min load {load1:.2f} > 2.0 -- the "
              f"single-thread protocol requires an idle machine.",
              file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rec = probe(a.dataset, a.object)
    # Disclosed, not assumed: the machine state this measurement was taken
    # under. Peak RSS is process-local and unaffected by other processes;
    # wall clock is not, so the load at start and end is recorded.
    rec["load_at_start_1min"] = round(load1, 2)
    rec["load_at_end_1min"] = round(os.getloadavg()[0], 2)
    out = OUT / f"{a.dataset}_{a.object}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"[ok] {out}  total={rec['wall_total_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
