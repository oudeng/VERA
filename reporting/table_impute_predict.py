"""Generate Table 5 (downstream impute-predict) as an `\\input`-able .tex (W3).

Layout of the published `tab:impute_predict_expanded`:

    Imputer | Model | AUROC | AUPRC | Accuracy | F1     (cells: mean +- std)

with panels per (dataset, task), separated by \\midrule, and `---` for metrics
a panel does not define. R1 differences the generator enforces rather than a
human remembering them:

  * all nine imputers appear (R0's Panel A silently lacked KNN/MICE/GAIN/MIWAE
    -- finding B7); a missing (imputer, model) combination in the input is an
    ERROR, not an omitted row;
  * the input comes from T2.4's within-fold protocol output.

Expected input schema (long CSV, written by experiments/downstream.py [[T2.4]]):
    panel, dataset, task, imputer, model, seed, metric, value
where metric in {AUROC, AUPRC, Accuracy, F1}.

    python reporting/table_impute_predict.py --long <T2.4 csv> \
        --out reporting/out/tab_impute_predict.tex
    python reporting/table_impute_predict.py --selftest   # format check only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

METRICS = ["AUROC", "AUPRC", "Accuracy", "F1"]
#: P5R-H SS7.2 (third review P1-2): rows are presented in protocol-class
#: blocks -- separable fit/transform methods first, then batch-transductive
#: methods -- and never ranked across the class boundary.
CLASS_BLOCKS = [("Inductive (separable fit/transform)",
                 ["MeanMode", "KNN", "MICE", "MissForest"]),
                ("Batch-transductive (per-block instances)",
                 ["GAIN", "MIWAE", "HyperImpute", "TabCSDI", "SNI"])]
IMPUTER_ORDER = [m for _t, ms in CLASS_BLOCKS for m in ms]
DISPLAY = {"MeanMode": "Mean/Mode"}


def _cell(vals: np.ndarray) -> str:
    if len(vals) == 0:
        return "---"
    return (rf"${np.mean(vals):.3f}{{\scriptstyle\pm{np.std(vals, ddof=1):.3f}}}$"
            if len(vals) > 1 else rf"${vals[0]:.3f}$")


def within_class_readout(long: "pd.DataFrame") -> dict:
    """P5R-K SS3 (fourth review SS5.2): rank and compare SNI ONLY inside its
    own protocol class.

    Fit/transform methods are inductive; the methods without that interface
    complete the test block with a separate instance, which makes their
    test-side imputation batch-transductive. Ranking across that boundary
    compares two deployment settings, so every ranking statement in the
    prose is computed here, within class, and never typed.
    """
    cls = {m: t for t, ms in CLASS_BLOCKS for m in ms}
    sni_class = [m for m in cls if cls[m] == cls["SNI"]]
    out = {}
    for panel in sorted(long.panel.unique()):
        g = long[long.panel == panel]
        for model in sorted(g.model.unique()):
            gm = g[(g.model == model) & (g.metric == "AUROC")]
            means = (gm[gm.imputer.isin(sni_class)]
                     .groupby("imputer").value.mean().sort_values(
                         ascending=False))
            order = list(means.index)
            best = order[0]
            per_seed = gm.pivot_table(index="seed", columns="imputer",
                                      values="value")
            d = per_seed["SNI"] - per_seed[best]
            out[f"{panel}|{model}"] = {
                "class": cls["SNI"], "n_in_class": len(sni_class),
                "sni_rank": order.index("SNI") + 1,
                "best_in_class": best,
                "sni_auroc": round(float(means["SNI"]), 4),
                "best_auroc": round(float(means[best]), 4),
                "paired_delta_mean": round(float(d.mean()), 4),
                "paired_delta_min": round(float(d.min()), 4),
                "paired_delta_max": round(float(d.max()), 4),
                "n_seeds": int(per_seed.shape[0])}
    return out


def build(long_path: Path, out_path: Path) -> Path:
    long = pd.read_csv(long_path)
    need = {"panel", "dataset", "task", "imputer", "model", "seed",
            "metric", "value"}
    if not need.issubset(long.columns):
        raise ValueError(f"input lacks columns {sorted(need - set(long.columns))}")

    rows, midrules = [], []
    panels = sorted(long.panel.unique())
    for panel in panels:
        g = long[long.panel == panel]
        ds, task = g.dataset.iloc[0], g.task.iloc[0]
        rows.append({"Imputer": rf"\multicolumn{{6}}{{l}}{{{{Panel {panel}: "
                                rf"{ds} ({task})}}}}",
                     "_span": True})
        for model in sorted(g.model.unique()):
            gm = g[g.model == model]
            present = set(gm.imputer.unique())
            missing = [m for m in IMPUTER_ORDER if m not in present]
            if missing:
                raise ValueError(
                    f"panel {panel} model {model}: imputers absent from the "
                    f"input: {missing}. B7 was exactly this omission published "
                    f"as 'omitted for table width'; refusing to emit a partial "
                    f"panel silently.")
            for blk_title, members in CLASS_BLOCKS:
                rows.append({"Imputer": rf"\multicolumn{{6}}{{l}}{{\emph{{"
                                        rf"{blk_title}}}}}",
                             "_span": True})
                for imp in members:
                    gi = gm[gm.imputer == imp]
                    row = {"Imputer": DISPLAY.get(imp, imp), "Model": model}
                    for met in METRICS:
                        row[met] = _cell(gi[gi.metric == met
                                            ].value.to_numpy(float))
                    rows.append(row)
            midrules.append(len(rows) - 1)
    midrules = midrules[:-1]  # no rule after the last block; bottomrule follows

    body = pd.DataFrame(rows).drop(columns=["_span"], errors="ignore").fillna("")
    style = TableStyle(
        # 45 body rows plus a fifteen-line note: at the class's default
        # spacing the block runs past the bottom margin and the folio
        # prints inside the note's last line (found by rendering page 29,
        # not by any LaTeX warning -- there is none for this).
        row_stretch=0.86,
        environment="table*",
        size=r"\scriptsize",
        notes=(
            r"\textit{Protocol:} no test-row information enters the "
            r"completed training block or the downstream predictor, and "
            r"the nine imputers reach that guarantee by two different "
            r"designs. The four with a separable fit/transform interface "
            r"-- mean/mode, KNN, MICE, MissForest -- fit on the training "
            r"block and transform the test block, which is inductive "
            r"imputation. The five without one -- GAIN, MIWAE, "
            r"HyperImpute, TabCSDI, SNI -- complete each block with a "
            r"separately constructed instance, so the test block is "
            r"completed jointly from its own observed entries, which is "
            r"batch-transductive imputation. Under both designs no "
            r"statistic of any test row reaches the completed training "
            r"table; the evaluated setting is batch secondary use, not "
            r"per-patient prospective deployment. Results are read within "
            r"protocol class and the two classes are never ranked against "
            r"each other. All nine "
            r"imputers are reported in every panel. Per seed, one "
            r"stratified 80/20 train/test split -- repeated holdout, not "
            r"cross-validation -- (stratified on the label; split seed = "
            r"the mask seed); the downstream model is fitted on the "
            r"training side only, and each imputer follows its own class's "
            r"design above. Rows are grouped by "
            r"protocol class and never ranked across the class boundary. "
            r"Five mask seeds "
            r"(1, 2, 3, 5, 8); 90 imputation units per panel family "
            r"(2 panels $\times$ 9 imputers $\times$ 5 seeds). Cells are "
            r"mean $\pm$ sample standard deviation over the five mask "
            r"seeds. Independence is verified per protocol class before "
            r"each batch by one representative method per class -- a "
            r"class-level sentinel, not a per-method audit: test-row "
            r"perturbation leaves the completed training "
            r"block bit-identical (both classes); training-row "
            r"perturbation changes the fit/transform methods' test "
            r"imputations (positive control) and leaves the "
            r"batch-transductive methods' test block bit-identical "
            r"($n_{\mathrm{changes}}=0$); the label has no entry point "
            r"into the imputation path with the split held fixed.",
            # src: docs/T44_downstream_rules.md (holdout clause) +
            #      evaluation verify_independence_per_class
            #      (results/T4_downstream/smoke_independence.json)
        ),
    )
    tex = dataframe_to_tex(
        body,
        caption=(r"Protocol-class-specific downstream analysis: task "
                 r"performance after imputation, reported within "
                 r"protocol classes and not ranked across them."),
        label="tab:impute_predict_expanded",
        column_format="llcccc",
        header=["Imputer", "Model", r"AUROC $\uparrow$", r"AUPRC $\uparrow$",
                r"Accuracy $\uparrow$", r"F1 $\uparrow$"],
        style=style,
        escape_data=False,
        midrule_after=midrules,
    )
    # span rows: dataframe_to_tex joins every column, leaving "& & &"
    # after a \multicolumn cell -- strip the empty tails.
    tex = re.sub(r'(\\multicolumn\{\d+\}\{l\}\{.*?\}\})(?:\s*&\s*)+(\\\\)',
                 r'\1 \2', tex)
    # Within-class readouts as macros + a machine-readable record, so the
    # prose cannot drift from the data or slip back across the class line.
    wc = within_class_readout(long)
    (out_path.parent.parent.parent / "results" / "T4_downstream"
     / "t44_within_class.json").write_text(json.dumps(wc, indent=1))
    lines = ["% generated by reporting/table_impute_predict.py -- within-"
             "protocol-class downstream readouts (P5R-K SS3)"]
    for key, v in wc.items():
        panel, model = key.split("|")
        tag = f"{panel}{model}"
        lines += [f"% {key}: SNI rank {v['sni_rank']}/{v['n_in_class']} in "
                  f"{v['class']}; best in class {v['best_in_class']}",
                  f"\\newcommand{{\\dsRank{tag}}}"
                  f"{{{v['sni_rank']} of {v['n_in_class']}}}",
                  f"\\newcommand{{\\dsBest{tag}}}"
                  f"{{{v['best_in_class'].replace('MeanMode', 'mean/mode')}}}",
                  # \mbox: a signed number must never break at its sign.
                  # It did -- "-" was left hanging at a line end on page 15,
                  # where it reads as a hyphen and the next line opens with a
                  # positive-looking "0.004".
                  # $...$: and the sign must be a MINUS. Outside math the same
                  # number set as "-0.008" prints a short hyphen beside the
                  # "$-0.030$" of every other effect in the paper (eighth-round
                  # freeze inspection, main p.15).
                  f"\\newcommand{{\\dsDelta{tag}}}"
                  f"{{\\mbox{{${v['paired_delta_mean']:+.3f}$}}}}"]
    (out_path.parent / "downstream_macros.tex").write_text(
        "\n".join(lines) + "\n")
    return write_tex(out_path, tex, provenance={
        "generator": "code_SNI/reporting/table_impute_predict.py",
        "input": str(long_path),
        "code_SNI commit": runconfig.git_commit(),
        "n_panels": len(panels),
    })


def _selftest() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for panel, ds, task, models in [("A", "MIMIC", "mortality, MAR 30\\%",
                                     ["LR", "XGB"]),
                                    ("B", "NHANES", "metabolic score, MAR 30\\%",
                                     ["pipeline"])]:
        for model in models:
            for imp in IMPUTER_ORDER:
                for seed in (1, 2, 3):
                    for met in METRICS:
                        if panel == "B" and met == "AUPRC":
                            continue        # exercises the --- path
                        rows.append({"panel": panel, "dataset": ds, "task": task,
                                     "imputer": imp, "model": model, "seed": seed,
                                     "metric": met,
                                     "value": rng.uniform(0.4, 0.99)})
    tmp = Path("/tmp") / "t5_selftest_long.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    out = build(tmp, CODE_ROOT / "reporting" / "out" / "tab_impute_predict_SELFTEST.tex")
    txt = out.read_text()
    assert txt.count("Panel") == 2 and "---" in txt and "Mean/Mode" in txt
    assert txt.count("SNI &") == 3, "one SNI row per (panel, model) block"
    print(f"[OK] selftest wrote {out} ({len(txt.splitlines())} lines)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long")
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_impute_predict.tex"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.long:
        ap.error("--long required (or --selftest)")
    out = build(Path(a.long), Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
