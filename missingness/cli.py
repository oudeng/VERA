"""Command line entry point for the missingness simulator.

Two subcommands:

``generate``
    Write masks for a dataset x mechanism x rate grid, driven entirely by
    ``configs/missingness.yaml`` and ``configs/datasets.yaml`` (principle E1).
    **This is for P2.** T1.5 ships it but generates nothing: the driver decision
    is pending task T1.6, so ``--profile`` has no default and the placeholder
    profile refuses to run without ``--i-know-this-is-a-placeholder``.

``diagnose``
    Read-only. Computes the audit numbers for a grid without writing any mask:
    per-column achieved rates, the row-index correlation, and the same figures
    for the ``record_index_ID`` legacy profile as a reference.

Examples::

    python -m missingness.cli diagnose --datasets eICU NHANES --rates 0.3 \\
        --profile clinical_v1_PLACEHOLDER --out results/T1.5_simulator

    # P2 only, after T1.6:
    python -m missingness.cli generate --datasets eICU --mechanisms MCAR MAR MNAR \\
        --rates 0.1 0.3 0.5 --profile clinical_v1 --out data/masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .generator import _rate_tag, generate, generate_and_write
from .spec import DEFAULT_DATASETS_CONFIG, load_config, load_datasets_config, resolve

_CODE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_table(dataset: str, explicit: Optional[Path]) -> Path:
    """Locate a dataset's complete table, preferring the code_SNI derived copy."""
    if explicit is not None:
        return Path(explicit)
    cfg = load_datasets_config()["datasets"][dataset]
    for key in ("complete_path", "source_path_R0"):
        p = cfg.get(key)
        if not p:
            continue
        cand = (_CODE_ROOT / p).resolve()
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    raise FileNotFoundError(
        f"no complete table found for {dataset}; declare `complete_path` in "
        f"{DEFAULT_DATASETS_CONFIG} or pass --input"
    )


def _grid(args) -> List[tuple]:
    datasets = args.datasets or list(load_datasets_config()["datasets"])
    return [(d, m, r) for d in datasets for m in args.mechanisms for r in args.rates]


def _overrides(args) -> dict:
    return {"row_order": {"mode": args.row_order, "seed": args.row_order_seed}}


def cmd_generate(args) -> int:
    profile = args.profile
    cfg = load_config(args.config)
    status = str((cfg["profiles"].get(profile) or {}).get("status", ""))
    if "PLACEHOLDER" in status and not args.i_know_this_is_a_placeholder:
        print(
            f"[REFUSED] profile {profile!r} is marked {status}. Its drivers are schema\n"
            f"          demonstrations, not the drivers we intend to publish; choosing\n"
            f"          them is a first-author decision pending task T1.6.\n"
            f"          Pass --i-know-this-is-a-placeholder to override.",
            file=sys.stderr,
        )
        return 2

    outdir = Path(args.out)
    for dataset, mechanism, rate in _grid(args):
        df = pd.read_csv(_resolve_table(dataset, args.input))
        spec = resolve(dataset, mechanism, rate, profile=profile,
                       config_path=args.config, seed=args.seed, overrides=_overrides(args))
        res = generate_and_write(df, spec, outdir / dataset)
        rates = res.meta["rates"]
        print(f"[OK] {dataset}_{mechanism}_{_rate_tag(rate)}  "
              f"eligible={rates['actual_rate_eligible']:.4f} "
              f"all={rates['actual_rate_all']:.4f} "
              f"row_index_r={res.meta['row_index_diagnostics']['pearson_r_rowrate_vs_rowindex_eligible']:+.4f} "
              f"E4={'ok' if res.meta['e4_mask_verification']['all_consistent'] else 'FAILED'}")
    return 0


def cmd_diagnose(args) -> int:
    rows: List[dict] = []
    percol: List[dict] = []
    for dataset, mechanism, rate in _grid(args):
        df = pd.read_csv(_resolve_table(dataset, args.input))
        for profile in (args.profile, "record_index_ID"):
            spec = resolve(dataset, mechanism, rate, profile=profile,
                           config_path=args.config, seed=args.seed,
                           overrides=_overrides(args))
            res = generate(df, spec)
            r = res.meta["row_index_diagnostics"]
            pc = res.meta["rates"]["per_column_missing_rate"]
            worst = max((abs(v - rate) for c, v in pc.items()
                         if c in set(spec.target_columns())), default=0.0)
            rows.append({
                "dataset": dataset, "mechanism": mechanism, "rate": rate,
                "profile": profile, "row_order": args.row_order, "n_rows": len(df),
                "actual_rate_eligible": res.meta["rates"]["actual_rate_eligible"],
                "actual_rate_all": res.meta["rates"]["actual_rate_all"],
                "worst_per_column_rate_error": worst,
                "n_columns_outside_tolerance":
                    len(res.meta["rates"]["columns_outside_tolerance"]),
                "row_index_pearson_r": r["pearson_r_rowrate_vs_rowindex_eligible"],
                "null_sd": 1.0 / np.sqrt(max(len(df) - 1, 1)),
            })
            for col, v in pc.items():
                percol.append({"dataset": dataset, "mechanism": mechanism, "rate": rate,
                               "profile": profile, "column": col, "missing_rate": v,
                               "masked": col in set(spec.target_columns())})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    suffix = "" if args.row_order == "as_is" else f"_{args.row_order}"
    summary.to_csv(out / f"diagnostics_summary{suffix}.csv", index=False)
    pd.DataFrame(percol).to_csv(out / f"diagnostics_per_column{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwritten: {out}/diagnostics_summary{suffix}.csv, "
          f"{out}/diagnostics_per_column{suffix}.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="missingness", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--datasets", nargs="*", default=None)
        sp.add_argument("--mechanisms", nargs="*", default=["MCAR", "MAR", "MNAR"])
        sp.add_argument("--rates", nargs="*", type=float, default=[0.1, 0.3, 0.5])
        sp.add_argument("--profile", required=True)
        sp.add_argument("--config", type=Path, default=None)
        sp.add_argument("--input", type=Path, default=None)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--row-order", choices=["as_is", "shuffle"], default="as_is",
                        help="'shuffle' randomises row order before masking, which removes "
                             "the confound from source tables that arrive sorted")
        sp.add_argument("--row-order-seed", type=int, default=20250728)

    g = sub.add_parser("generate", help="write masks (P2)")
    common(g)
    g.add_argument("--out", required=True)
    g.add_argument("--i-know-this-is-a-placeholder", action="store_true")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("diagnose", help="audit a grid without writing masks")
    common(d)
    d.add_argument("--out", default="results/T1.5_simulator")
    d.set_defaults(func=cmd_diagnose)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
