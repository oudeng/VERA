"""W4 -- the ESM's deterministic sections, generated from the config truth.

Four fragments, all `\\input`-able, none hand-written:

  tab_sm_datasets.tex       Table S1 from configs/datasets.yaml
  tab_hparams.tex           S3.5 hyperparameters from configs/methods.yaml
                            (B1's structural fix: the table that contradicted
                            every manifest is now emitted from the file the
                            runners execute)
  sec_missingness_spec.tex  S3.2 mechanism specification from
                            configs/missingness.yaml -- drivers, coefficients
                            and the per-block rationales verbatim
  tab_runtime_frame.tex     S6 runtime table FRAME: per-method device (from
                            scheduling.yaml, B2's fix) and thread policy now,
                            timing cells emitted as [[GRID]] until the grid
                            lands. No provisional numbers, ever.

    PYTHONHASHSEED=2025 python reporting/esm_sections.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import (display_name, TableStyle, dataframe_to_tex,      # noqa: E402
                             code_cell, escape_cell, write_tex)

OUT = CODE_ROOT / "reporting" / "out"
_PROV = {"code_SNI commit": runconfig.git_commit()}


def _cfg(name: str) -> dict:
    return yaml.safe_load((CODE_ROOT / "configs" / f"{name}.yaml").read_text())


# --------------------------------------------------------------------------- #
def gen_datasets() -> Path:
    ds_cfg = _cfg("datasets")["datasets"]
    rows = []
    for name, blk in ds_cfg.items():
        cols = blk.get("columns", {})
        roles = {c: v.get("role") for c, v in cols.items()}
        feats = [c for c, r in roles.items() if r in ("imputable", "always_observed")]
        cats = [c for c in feats if cols[c].get("type") == "categorical"]
        drivers = blk.get("always_observed", [])
        rows.append({
            "Dataset": name,
            "n": blk["n_rows"],
            "d": len(feats),
            "imp": blk["n_imputable"],
            "catcont": f"{len(cats)}/{len(feats) - len(cats)}",
            "drivers": str(len(drivers)),
            "target": code_cell(blk.get("downstream_target", "---")),
            "cov": f"{100 * float(blk['evaluation_coverage']):.0f}\\%",
        })
    body = pd.DataFrame(rows)
    style = TableStyle(environment="table", notes=(
        r"\textit{Notes:} $d$ counts model features; the identifier and the "
        r"downstream target are never visible to any imputer. Always-observed "
        r"features are the MAR drivers (Online Resource~1, "
        r"\emph{Missingness mechanism specification}) and are "
        r"excluded from evaluation coverage. Table construction and per-column "
        r"roles are specified in \texttt{configs/datasets.yaml}, from which this "
        r"table is generated.",))
    tex = dataframe_to_tex(
        body,
        caption=r"Datasets after the rebuild for this revision.",
        label="tab:sm_datasets",
        header=["Dataset", "$n$", "$d$", "imputable", "cat/cont",
                "drivers", "downstream target", "eval.\\ cov."],
        style=style, escape_data=False)
    out = write_tex(OUT / "tab_sm_datasets.tex", tex, provenance={
        **_PROV, "generator": "reporting/esm_sections.py --datasets",
        "input": "configs/datasets.yaml"})

    # Main-text variant: the full table overflows a two-column measure
    # (P4-C revision 3), so the main text gets a spanning table* with merged
    # columns; drivers and coverage move into the note. Same source, same
    # generator, no hand-edited numbers.
    main = body[["Dataset", "n"]].copy()
    main["dimp"] = body.apply(lambda r: rf"{r['d']} ({r['imp']})", axis=1)
    main["catcont"] = body["catcont"]
    main["target"] = body["target"]
    cov = "; ".join(rf"{r.Dataset} {r.cov}" for r in body.itertuples())
    main_style = TableStyle(environment="table*", notes=(
        r"\textit{Notes:} $d$ counts model features (imputable in parentheses); "
        r"the identifier and the downstream target are never visible to any "
        r"imputer. Always-observed features serve as MAR drivers "
        r"(Online Resource~1, \emph{Missingness mechanism specification}) "
        r"and are excluded from evaluation; "
        r"the resulting evaluation coverage is " + cov + r". Generated from "
        r"\texttt{configs/datasets.yaml}.",))
    main_tex = dataframe_to_tex(
        main,
        caption=r"Datasets after the rebuild for this revision.",
        label="tab:datasets_main",
        header=["Dataset", "$n$", "$d$ (imp.)", "cat/cont",
                "downstream target"],
        style=main_style, escape_data=False)
    write_tex(OUT / "tab_datasets_main.tex", main_tex, provenance={
        **_PROV, "generator": "reporting/esm_sections.py --datasets (main variant)",
        "input": "configs/datasets.yaml"})
    return out


# --------------------------------------------------------------------------- #
_SNI_GROUPS = [
    ("EM outer loop", ["max_iters", "alpha0", "gamma", "tol", "use_stat_refine",
                       "mask_fraction"]),
    ("CPFA architecture", ["emb_dim", "num_heads", "hidden_dims"]),
    ("Optimization", ["lr", "epochs", "batch_size", "weight_decay"]),
    ("Prior strength", ["lambda_mode", "lambda_fixed_value"]),
]


def gen_hparams() -> Path:
    m = _cfg("methods")
    sni = m["sni"]
    rows, rules = [], []
    for group, keys in _SNI_GROUPS:
        rows.append({"Parameter": rf"\textit{{{group}}}", "Value": ""})
        for k in keys:
            rows.append({"Parameter": rf"\texttt{{{escape_cell(k)}}}",
                         "Value": escape_cell(sni[k])})
        rules.append(len(rows) - 1)
    sni_tex = dataframe_to_tex(
        pd.DataFrame(rows),
        caption=(r"SNI configuration (this revision). Generated from "
                 r"\texttt{configs/methods.yaml}, the file the runners execute; "
                 r"the original submission's supplement's hand-written values disagreed with every "
                 r"recorded manifest and are corrected here."),
        label="tab:hparams_sni",
        column_format="ll",
        header=["Parameter", "Value"],
        style=TableStyle(environment="table"),
        escape_data=False,
        midrule_after=rules[:-1])

    brows = []
    for name, blk in m["baselines"].items():
        params = {k: v for k, v in blk.items() if not k.startswith("_")}
        ptxt = ", ".join(f"{escape_cell(k)}={escape_cell(v)}"
                         for k, v in params.items()) or "---"
        brows.append({"Method": display_name(name), "Hyperparameters": ptxt,
                      "Source": escape_cell(blk.get("_params_source", ""))})
    base_tex = dataframe_to_tex(
        pd.DataFrame(brows),
        caption=r"Baseline hyperparameters (this revision), generated from the same file.",
        label="tab:hparams_baselines",
        column_format=(r"l>{\raggedright\arraybackslash}p{0.55\linewidth}"
                       r">{\raggedright\arraybackslash}p{0.25\linewidth}"),
        header=["Method", "Hyperparameters", "Source"],
        # "Mean/Mode" is one character wider than the key it replaced, and
        # this table had no slack; 4 pt of column padding pays for it, which
        # is cheaper than putting the key back.
        style=TableStyle(environment="table", col_sep_pt=4.0),
        escape_data=False)
    return write_tex(OUT / "tab_hparams.tex", sni_tex + "\n" + base_tex,
                     provenance={**_PROV,
                                 "generator": "reporting/esm_sections.py --hparams",
                                 "input": "configs/methods.yaml"})


# --------------------------------------------------------------------------- #
def _coef_str(c, code: bool = False, depth: int = 0) -> str:
    """Render a specification for a reader, never Python's repr of it.

    The old version handled a top-level dict and let everything else fall
    through to str(): a list of driver names printed as ["gcs", "age_years"]
    with its brackets and quotes, and a per-level dict printed as {0: 0.5,
    1: -0.8} with its braces -- in the Coefficients column, so the leak
    landed on numbers. reporting/latex.refuse_repr now refuses both shapes at
    the emission layer; this is what it forces.
    """
    if isinstance(c, dict):
        # Depth-0 keys are driver COLUMN names, so they are code; deeper keys
        # are the driver's levels (0, 1), which are not. Without the split
        # "gcs" printed in roman beside "age\_years" in typewriter, in one
        # cell (eighth-round freeze inspection, ESM p. 6).
        key = code_cell if depth == 0 else escape_cell
        return "; ".join(f"{key(k)}: {_coef_str(v, code, depth + 1)}"
                         for k, v in c.items())
    if isinstance(c, (list, tuple)):
        return ", ".join(_coef_str(v, code, depth) for v in c)
    return code_cell(c) if code else escape_cell(c)


def gen_missingness() -> Path:
    prof = _cfg("missingness")["profiles"]["clinical_v1"]
    parts = [r"% S3.2 mechanism specification, generated from "
             r"configs/missingness.yaml (profile clinical_v1)."]
    for ds, blk in prof["datasets"].items():
        parts.append(rf"\subsubsection*{{{escape_cell(ds)}}}")
        ao = blk.get("common", {}).get("always_observed", [])
        parts.append(r"Always observed (drivers): "
                     + ", ".join(rf"\texttt{{{escape_cell(c)}}}" for c in ao)
                     + r".\par")
        for mech in ("MCAR", "MAR", "MNAR"):
            mblk = blk.get(mech)
            if not mblk:
                continue
            parts.append(rf"\paragraph{{{mech}}}")
            rat = " ".join(str(mblk.get("rationale", "")).split())
            parts.append(escape_cell(rat) + r"\par")
            spec = mblk.get("mar") if mech == "MAR" else mblk.get("mnar")
            if not spec:
                continue
            rows = []
            default = spec.get("default", {})
            if default:
                rows.append({"Column": r"\textit{default}",
                             "Drivers / mode": (
                                 _coef_str(default["drivers"], code=True)
                                 if "drivers" in default
                                 else _coef_str(default.get("mode", ""))),
                             "Coefficients": _coef_str(
                                 default.get("coefficients",
                                             default.get("coefficient", "")))})
            for col, cs in (spec.get("columns") or {}).items():
                rows.append({"Column": rf"\texttt{{{escape_cell(col)}}}",
                             "Drivers / mode": (
                                 _coef_str(cs["drivers"], code=True)
                                 if "drivers" in cs
                                 else _coef_str(cs.get("mode", ""))),
                             "Coefficients": _coef_str(
                                 cs.get("coefficients", cs.get("coefficient", "")))})
            tex = dataframe_to_tex(
                pd.DataFrame(rows),
                caption=rf"{escape_cell(ds)} {mech} specification.",
                label=f"tab:mech_{ds}_{mech}".lower(),
                column_format=(r"l>{\raggedright\arraybackslash}p{0.35\linewidth}"
                               r">{\raggedright\arraybackslash}p{0.35\linewidth}"),
                header=["Column", "Drivers / mode", "Coefficients"],
                style=TableStyle(environment="table"),
                escape_data=False)
            parts.append(tex)
    return write_tex(OUT / "sec_missingness_spec.tex",
                     "% line-breaking elasticity only; zero semantic change (P5 SS3.6)\n"
                     "{\\emergencystretch=1em\n"
                     + "\n".join(parts) + "\n}\n",
                     provenance={**_PROV,
                                 "generator": "reporting/esm_sections.py --missingness",
                                 "input": "configs/missingness.yaml profile clinical_v1"})


# --------------------------------------------------------------------------- #
def _secs(v: float) -> str:
    """Median seconds per run, never printed as a bare rounded zero."""
    return "$<1$" if v < 0.5 else f"{v:,.0f}"


def runtime_rows() -> list[dict]:
    """Device column of the S6 runtime table.

    Reads configs/scheduling.yaml `method_placement` -- the binding source
    under ruling A-2 -- and deliberately nothing from methods.yaml, which
    P3-A ruling 1 stripped of its historical `device:` block precisely so
    this generator cannot print an R0 record as an R1 declaration.
    Asserted in tests/test_adjudications.py.
    """
    placement = _cfg("scheduling")["method_placement"]
    grid = CODE_ROOT / "results" / "P2_main_grid"
    med = {}
    if grid.exists():
        # Same records as the cost table's fit column (registry C-6): the
        # grid's own recorded wall clocks, median per method over all its
        # completed cells.
        from stats.aggregate_grid import load_runs
        runs = load_runs(grid)
        med = runs.groupby("method").runtime_sec.median().to_dict()
    return [{"Method": display_name(m), "Device": dev.upper(),
             "Threads": "2 (pinned, verified)" if dev == "cpu" else "---",
             # A sub-second median rounds to "0", which reads as "took no
             # time" rather than "faster than the reporting precision"
             # (P5R-K SS4.2, page inspection).
             "sec": (_secs(med[m]) if m in med else "[[GRID]]")}
            for m, dev in placement.items()]


def gen_runtime_frame() -> Path:
    rows = runtime_rows()
    style = TableStyle(environment="table", notes=(
        r"\textit{Notes:} devices are declared per method in "
        r"\texttt{configs/scheduling.yaml} and recorded per run; SNI's figures "
        r"are device- and thread-count-specific, so the reproduction "
        r"recipe in this section pins both. Timing cells are filled by the "
        r"reporting pipeline from the grid's recorded wall clocks; this frame "
        r"carries no provisional numbers.",))
    tex = dataframe_to_tex(
        pd.DataFrame(rows),
        caption=r"Execution environment and median per-run wall clock.",
        label="tab:runtime",
        header=["Method", "Device", "BLAS threads", r"median s/run (grid)"],
        style=style, escape_data=False)
    return write_tex(OUT / "tab_runtime_frame.tex", tex, provenance={
        **_PROV, "generator": "reporting/esm_sections.py --runtime",
        "input": "configs/scheduling.yaml"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", action="store_true")
    ap.add_argument("--hparams", action="store_true")
    ap.add_argument("--missingness", action="store_true")
    ap.add_argument("--runtime", action="store_true")
    a = ap.parse_args()
    todo = []
    if a.all or a.datasets:
        todo.append(gen_datasets)
    if a.all or a.hparams:
        todo.append(gen_hparams)
    if a.all or a.missingness:
        todo.append(gen_missingness)
    if a.all or a.runtime:
        todo.append(gen_runtime_frame)
    if not todo:
        ap.error("nothing to do; pass --all or a section flag")
    for fn in todo:
        print(f"[OK] wrote {fn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
