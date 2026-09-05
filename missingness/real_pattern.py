"""T2b.2: an evaluation condition built from a REAL missingness pattern.

Every other condition in this benchmark masks a complete table with a mechanism
we wrote. This one masks it with a pattern nobody wrote: the row-level
missingness actually observed in `heart_2022_with_nans.csv` (445,132 BRFSS
respondents), resampled onto the complete table so that ground truth is known.

It exists because the T2.2(d) comparison exposed two limits of *any* simulated
mechanism, ours included:

* real per-column rates span 0.00-18.54 %, while a per-column-calibrated mask
  gives every column the same rate by construction (measured sd 0.0000);
* real missingness is clustered -- per-row missing counts carry 10.91x the
  variance of an independent Bernoulli mask, against 2.70x for our MAR and
  1.00-1.02x for MCAR/MNAR.

Rather than report those as limitations, this turns them into a condition. The
resampling is at **row level**, which is what preserves all three properties at
once: the per-column rate profile, the co-occurrence structure, and derived-
variable propagation (P(BMI missing | height or weight missing) = 1.0000 in the
real data, and an independent per-column simulation cannot express it).

**The honest boundary, which the report must state**: this is survey item
non-response from a telephone health survey. It is not an EHR missingness
mechanism, and nothing here should be extrapolated to one.
"""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
#: CDC2022 is a public download but not redistributed here; point at it
#: with CDC2022_DIR. The default is repo-relative, not a home directory.
CDC_DIR = Path(os.environ.get(
    "CDC2022_DIR", str(Path(__file__).resolve().parents[2]
                       / "data_CDC2022")))


def draw_real_patterns(n: int, seed: int,
                       real_path: Path = CDC_DIR / "heart_2022_with_nans.csv",
                       columns: Optional[Sequence[str]] = None,
                       always_observed: Sequence[str] = ()) -> pd.DataFrame:
    """Sample `n` whole rows of the observed missingness indicator, with
    replacement.

    Whole rows, not per-column draws: the point is the joint structure. Sampling
    each column independently would reproduce the marginal rates and destroy
    exactly the clustering this condition exists to capture.

    Columns in `always_observed` are forced observed afterwards, so the condition
    stays comparable with the simulated ones, which never mask a driver.
    """
    real = pd.read_csv(real_path)
    cols = list(columns) if columns else list(real.columns)
    na = real[[c for c in cols if c in real.columns]].isna()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(na), size=n)
    out = na.iloc[idx].reset_index(drop=True)
    for c in always_observed:
        if c in out.columns:
            out[c] = False
    return out


def apply_to(complete: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask aligned to `complete`'s columns; unmatched columns stay observed."""
    mask = pd.DataFrame(False, index=complete.index, columns=complete.columns)
    for c in complete.columns:
        if c in patterns.columns:
            mask[c] = patterns[c].to_numpy()
    return mask


def describe(mask: pd.DataFrame, targets: Sequence[str]) -> dict:
    t = list(targets)
    arr = mask[t].to_numpy(bool)
    rates = arr.mean(axis=0)
    cnt = arr.sum(axis=1).astype(float)
    var_null = float((rates * (1 - rates)).sum())
    return {
        "n_rows": int(len(mask)), "n_target_columns": len(t),
        "overall_rate": float(arr.mean()),
        "col_rate_min": float(rates.min()), "col_rate_max": float(rates.max()),
        "col_rate_sd": float(rates.std(ddof=0)),
        "row_count_var": float(cnt.var(ddof=0)),
        "row_count_var_independent_null": var_null,
        "dispersion_ratio": (float(cnt.var(ddof=0) / var_null)
                             if var_null > 0 else float("nan")),
        "frac_rows_fully_observed": float((cnt == 0).mean()),
        "r_rowrate_vs_rowindex": (
            float(np.corrcoef(arr.mean(axis=1), np.arange(len(arr)))[0, 1])
            if arr.mean(axis=1).std() > 0 else 0.0),
    }


def main() -> int:
    import yaml
    from common.rowspace import fingerprint
    from common import masks as common_masks

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CDC2022")
    ap.add_argument("--table-dir", default=str(ROOT / "data" / "derived_shuffled"))
    ap.add_argument("--out", default=str(ROOT / "data" / "masks" / "real_pattern"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 5, 8])
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())["datasets"]
    blk = cfg[a.dataset]
    ident = blk.get("identifier_column", "ID")
    target = blk.get("downstream_target")
    ao = list(blk.get("always_observed", []) or []) + [ident]

    complete = pd.read_csv(Path(a.table_dir) / f"{a.dataset}_complete.csv")
    targets = [c for c in complete.columns if c not in set(ao) | {target}]

    outdir = Path(a.out) / a.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        pat = draw_real_patterns(len(complete), seed,
                                 columns=list(complete.columns), always_observed=ao)
        mask = apply_to(complete, pat)
        stem = f"{a.dataset}_REALPATTERN_s{seed}"

        np.save(outdir / f"{stem}_mask.npy", mask.to_numpy().astype(np.uint8))
        X_missing = complete.mask(mask)
        X_missing.to_csv(outdir / f"{stem}.csv", index=False)

        # E4: reload and assert, exactly as the simulated masks do.
        reloaded = pd.read_csv(outdir / f"{stem}.csv")
        _, chk = common_masks.load_and_verify(
            reloaded, outdir / f"{stem}_mask.npy",
            columns=list(complete.columns), strict=True)

        stats = describe(mask, targets)
        meta = {
            "schema_version": 1,
            "generator": "missingness/real_pattern.py",
            "condition": "REAL_PATTERN",
            "source": str(CDC_DIR / "heart_2022_with_nans.csv"),
            "resampling": "whole rows of the observed missingness indicator, "
                          "with replacement -- preserves per-column rates, "
                          "co-occurrence and derived-variable propagation",
            "spec": {"dataset": a.dataset, "mechanism": "REAL_PATTERN",
                     "seed": seed, "target_columns": targets,
                     "observed_columns": ao},
            "shape": {"n_rows": int(len(complete)), "n_cols": int(complete.shape[1])},
            "columns": list(complete.columns),
            "rates": {"per_column_missing_rate":
                      {c: float(mask[c].mean()) for c in complete.columns},
                      "columns_outside_tolerance": []},
            "statistics": stats,
            "row_order": {"mode": "inherited_from_table", "seed": None,
                          "permutation": None,
                          "rowspace_digest": fingerprint(complete[ident].tolist())},
            "e4_mask_verification": {"all_consistent": bool(chk.consistent)},
            "boundary": "BRFSS telephone-survey item non-response. NOT an EHR "
                        "missingness mechanism; do not extrapolate.",
        }
        (outdir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        rows.append({"seed": seed, "E4": chk.consistent, **stats})
        print(f"[OK] {stem}  rate={stats['overall_rate']:.4f} "
              f"col_sd={stats['col_rate_sd']:.4f} "
              f"dispersion={stats['dispersion_ratio']:.2f}x "
              f"fully_observed={stats['frac_rows_fully_observed']*100:.1f}% "
              f"E4={'ok' if chk.consistent else 'FAILED'}")

    df = pd.DataFrame(rows)
    res = ROOT / "results" / "T2b_real_pattern"
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "real_pattern_masks.csv", index=False)
    print(f"\n{len(df)} masks; mean dispersion {df.dispersion_ratio.mean():.2f}x, "
          f"col-rate sd {df.col_rate_sd.mean():.4f}")
    print(f"wrote {outdir} and {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
