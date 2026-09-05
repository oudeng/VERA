"""T2.0: turn the two gate runs into an explicit written verdict.

Gate (a) asks whether the ported SNI is the same program as R0's. The criterion
in the P2 instruction is that the two sides agree bit for bit, or differ by less
than 1e-6. This script does not summarize the agreement -- it enumerates every
comparison and reports the worst one, because an average over 10 runs would hide
a single divergent seed, which is exactly the failure mode P1 was written to
prevent.

Three artifacts are compared per run, deliberately at different levels:

* `metrics_summary.json` -- the numbers that reach the paper.
* `dependency_matrix.csv` -- md5. This is the object reviewer point R2-1 is
  about, so "the metrics match" is not sufficient; the audit output has to match
  too.
* `imputed.csv` -- md5, written at %.17g so the artifact is not what rounds.
  This is the strictest of the three: two runs can agree on every reported metric
  and still differ somewhere in the matrix.

Gate (b) asks whether TabCSDI on the GPU reproduces R0's published values. Here
the criterion is looser by design -- agreement within R0's own cross-seed spread
-- because TabCSDI is stochastic and R0 fixed no seed for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "results" / "T2.0_gate"

#: P2 T2.0(a).
TOL_A = 1e-6

#: Fields in metrics_summary.json that are *not* results and must never enter the
#: equivalence comparison. Wall-clock will never match between two runs, and
#: including it made the first draft of this script report a "worst metric
#: difference" of 9.6 -- which was 9.6 seconds of runtime, not a numerical
#: divergence. `hash_probe` is retained deliberately: it is the B48 witness and
#: must be equal, so it is compared, not excluded.
NON_METRIC_KEYS = {"runtime_sec", "runtime_sec_wall", "rep"}


def md5(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _flat(d: dict, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def gate_a(root: Path = GATE) -> dict:
    r0_dir, port_dir = root / "cpu_r0", root / "cpu_port"
    rows: List[dict] = []
    missing: List[str] = []

    runs = sorted({p.name for p in r0_dir.glob("*") if p.is_dir()}
                  | {p.name for p in port_dir.glob("*") if p.is_dir()})
    for run in runs:
        a, b = r0_dir / run, port_dir / run
        if not (a / "metrics_summary.json").exists() or \
           not (b / "metrics_summary.json").exists():
            # Never silently drop an incomplete pair: that is finding B2's
            # failure mode (300 runs excluded by a comment in an aggregator).
            missing.append(run)
            continue

        ma = {k: v for k, v in
              _flat(json.loads((a / "metrics_summary.json").read_text())).items()
              if k not in NON_METRIC_KEYS}
        mb = {k: v for k, v in
              _flat(json.loads((b / "metrics_summary.json").read_text())).items()
              if k not in NON_METRIC_KEYS}
        shared = sorted(set(ma) & set(mb))
        worst_k, worst_v = None, 0.0
        n_exact = 0
        for k in shared:
            x, y = ma[k], mb[k]
            if x == y:
                n_exact += 1
                continue
            if math.isnan(x) and math.isnan(y):
                n_exact += 1
                continue
            d = abs(x - y)
            if d > worst_v:
                worst_k, worst_v = k, d

        rows.append({
            "run": run,
            "n_metrics": len(shared),
            "n_bit_identical": n_exact,
            "worst_metric": worst_k,
            "worst_abs_diff": worst_v,
            "only_in_r0": sorted(set(ma) - set(mb)),
            "only_in_port": sorted(set(mb) - set(ma)),
            "md5_D_r0": md5(a / "dependency_matrix.csv"),
            "md5_D_port": md5(b / "dependency_matrix.csv"),
            "md5_imputed_r0": md5(a / "imputed.csv"),
            "md5_imputed_port": md5(b / "imputed.csv"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"gate": "a", "verdict": "INCOMPLETE", "n_pairs": 0,
                "missing_runs": missing}

    df["D_match"] = df.md5_D_r0 == df.md5_D_port
    df["imputed_match"] = df.md5_imputed_r0 == df.md5_imputed_port

    worst = float(df.worst_abs_diff.max())
    all_metrics_exact = bool((df.n_bit_identical == df.n_metrics).all())
    schema_clean = bool(df.only_in_r0.map(len).sum() == 0
                        and df.only_in_port.map(len).sum() == 0)
    verdict = "PASS" if (worst < TOL_A and df.D_match.all()
                         and df.imputed_match.all() and schema_clean
                         and not missing) else "FAIL"

    return {
        "gate": "a",
        "criterion": f"every metric bit-identical, or |diff| < {TOL_A:g}; "
                     f"dependency_matrix.csv and imputed.csv md5 equal on both sides",
        "verdict": verdict,
        "n_pairs": int(len(df)),
        "missing_runs": missing,
        "all_metrics_bit_identical": all_metrics_exact,
        "worst_abs_diff": worst,
        "worst_run": (df.loc[df.worst_abs_diff.idxmax(), "run"]
                      if worst > 0 else None),
        "n_D_matching": int(df.D_match.sum()),
        "n_imputed_matching": int(df.imputed_match.sum()),
        "metric_schema_identical": schema_clean,
        "table": df,
    }


#: R0's own per-run record. Comparing against this *per seed* is far stronger
#: than the "within R0's cross-seed spread" criterion the instruction allows:
#: on NHANES that spread runs from R2 = +0.16 to -224, so almost anything would
#: fall inside it. Per-seed agreement leaves no such room.
#: P7-A closeout: no private absolute path in a published file. The
#: R0 tree is not in this repository (it holds restricted derived
#: tables); point at it with SNI_R0_ROOT, and default to the sibling
#: directory a full checkout would have. A clone that lacks it gets a
#: path it can act on rather than a stranger's home directory.
R0_ROOT = Path(os.environ.get("SNI_R0_ROOT",
                    Path(__file__).resolve().parents[2]
                    / "project_sni_R0"))
R0_SUMMARY = R0_ROOT / "results_all" / "agg_baselines_new/summary_all.csv"

GATE_B_METRICS = ["cont_NRMSE", "cont_RMSE", "cont_MAE", "cont_MB", "cont_R2",
                  "cont_Spearman", "cat_Accuracy", "cat_Macro-F1",
                  "cat_Cohen_kappa"]


def gate_b(root: Path = GATE, mechanism: str = "MAR",
           rate_tag: str = "30per") -> dict:
    """Compare each TabCSDI GPU run against R0's record for the same seed."""
    summary = root / "tabcsdi" / "summary_gpu_legacy.csv"
    if not summary.exists():
        return {"gate": "b", "verdict": "INCOMPLETE", "n_runs": 0,
                "reason": f"{summary} not written yet"}
    ours = pd.read_csv(summary)

    if not R0_SUMMARY.exists():
        return {"gate": "b", "verdict": "INCOMPLETE", "n_runs": int(len(ours)),
                "reason": f"{R0_SUMMARY} missing; cannot compare"}
    r0 = pd.read_csv(R0_SUMMARY)
    r0 = r0[(r0.algo == "TabCSDI") & (r0.mechanism == mechanism)
            & (r0.rate == rate_tag)]

    rows = []
    for _, o in ours.iterrows():
        run = f"{o.dataset}_s{int(o.seed)}"
        m = r0[(r0.dataset == o.dataset) & (r0.seed == o.seed)]
        if len(m) != 1:
            # Listed, never dropped.
            rows.append({"run": run, "status": f"R0 has {len(m)} matching rows",
                         "n_bit_identical": 0, "worst_abs_diff": float("nan")})
            continue
        m = m.iloc[0]
        diffs = {k: abs(float(o[k]) - float(m[k])) for k in GATE_B_METRICS
                 if k in o.index and k in m.index}
        worst_k = max(diffs, key=diffs.get) if diffs else None
        rows.append({
            "run": run, "status": "ok",
            "n_metrics": len(diffs),
            "n_bit_identical": sum(1 for v in diffs.values() if v == 0.0),
            "worst_metric": worst_k,
            "worst_abs_diff": diffs[worst_k] if worst_k else float("nan"),
            "ours_R2": float(o["cont_R2"]), "r0_R2": float(m["cont_R2"]),
            "runtime_ours": float(o.get("runtime_sec", float("nan"))),
            "runtime_r0": float(m.get("runtime_sec", float("nan"))),
        })

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    verdict = "FAIL"
    if len(ok) == len(df) and len(df) > 0:
        verdict = ("PASS" if float(ok.worst_abs_diff.max()) < TOL_A else "FAIL")

    out = {"gate": "b",
           "criterion": "every metric of every run reproduces R0's own recorded "
                        "value for the same seed to better than 1e-6",
           "verdict": verdict, "n_runs": int(len(df)),
           "n_comparable": int(len(ok)),
           "n_runs_all_metrics_identical": int((ok.n_bit_identical
                                                == ok.n_metrics).sum())
           if len(ok) else 0,
           "worst_abs_diff": float(ok.worst_abs_diff.max()) if len(ok) else None,
           "table": df}

    # The divergence finding (B70) is recorded here rather than left for a
    # reader to spot: reproducing a diverged run faithfully is a pass for this
    # gate and a problem for the paper, and those are two different statements.
    if len(ok):
        bad = ok[ok.ours_R2 < 0]
        out["runs_with_negative_R2"] = int(len(bad))
        out["worst_R2"] = float(ok.ours_R2.min())
        out["divergence_note"] = (
            f"{len(bad)}/{len(ok)} runs have R2 < 0, worst {ok.ours_R2.min():.2f}. "
            f"These reproduce R0 exactly, so the gate passes; see finding B70 for "
            f"what it means for the published TabCSDI row.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(GATE))
    ap.add_argument("--out", default=str(GATE / "verdict.json"))
    a = ap.parse_args()
    root = Path(a.root)

    ra = gate_a(root)
    print("=" * 74)
    print(f"GATE (a) equivalence -- verdict: {ra['verdict']}   "
          f"pairs: {ra['n_pairs']}")
    print("=" * 74)
    if ra.get("missing_runs"):
        print(f"  INCOMPLETE PAIRS (listed individually, never skipped): "
              f"{ra['missing_runs']}")
    if ra["n_pairs"]:
        t = ra["table"]
        print(f"  metrics bit-identical in every run : {ra['all_metrics_bit_identical']}")
        print(f"  worst |diff| over all metrics      : {ra['worst_abs_diff']:.3e}"
              + (f"  (run {ra['worst_run']})" if ra["worst_run"] else ""))
        print(f"  dependency_matrix.csv md5 matches  : "
              f"{ra['n_D_matching']}/{ra['n_pairs']}")
        print(f"  imputed.csv md5 matches            : "
              f"{ra['n_imputed_matching']}/{ra['n_pairs']}")
        print()
        print(t[["run", "n_metrics", "n_bit_identical", "worst_abs_diff",
                 "D_match", "imputed_match"]].to_string(index=False))
        t.to_csv(root / "gate_a_pairs.csv", index=False)

    rb = gate_b(root)
    print()
    print("=" * 74)
    print(f"GATE (b) TabCSDI reproduction -- verdict: {rb['verdict']}   "
          f"runs: {rb['n_runs']}")
    print("=" * 74)
    if rb.get("table") is not None and not rb["table"].empty:
        print(f"  comparable against R0's record     : "
              f"{rb['n_comparable']}/{rb['n_runs']}")
        print(f"  runs with every metric identical   : "
              f"{rb['n_runs_all_metrics_identical']}/{rb['n_comparable']}")
        if rb.get("worst_abs_diff") is not None:
            print(f"  worst |diff| vs R0                 : "
                  f"{rb['worst_abs_diff']:.3e}")
        if rb.get("divergence_note"):
            print(f"  NOTE (B70): {rb['divergence_note']}")
        print()
        print(rb["table"].to_string(index=False))
        rb["table"].to_csv(root / "gate_b_runs.csv", index=False)

    payload = {k: v for k, v in ra.items() if k != "table"}
    payload_b = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                 for k, v in rb.items()}
    Path(a.out).write_text(json.dumps({"gate_a": payload, "gate_b": payload_b},
                                      indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
