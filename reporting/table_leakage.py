"""Leakage-axis table (P4O SS3): six objects x seven injected conditions.

Source: results/T4_leakage/t42_summary.json (the analyze-stage output).
Cells are detection counts out of 6 (2 datasets x 3 seeds) under
threshold-calibrated proxy discrimination; the discrepancy-control column
reads under both estimands (host-use fidelity vs risk screening). The
permutation-null detection rate is printed per object; objects whose null
rate exceeds nominal alpha are named in the note by computation, never by
hand (P4O: 0.143 / 0.095 must be visible). Historical class keys of the
frozen artifacts are rendered through reporting/termmap.py.

Guards: all six objects and all seven conditions present or refusal;
counts and n are read, never derived here.

    PYTHONHASHSEED=2025 python reporting/table_leakage.py
    PYTHONHASHSEED=2025 python reporting/table_leakage.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.disclosures import INFO_ASYMMETRY_LEAKAGE
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

T42 = CODE_ROOT / "results" / "T4_leakage" / "t42_summary.json"
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest", "Permutation-on-SNI"]
from reporting.termmap import condition_order  # noqa: E402
CONDS = condition_order()
HEADS = ["Exact", r"N$\rho$.95", r"N$\rho$.80",
         r"N$\rho$.60", "Interact.", "Conseq.",
         r"Discr.\ ctrl"]
DISPLAY = {"P": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           "Permutation-on-SNI": "Permutation-on-SNI",
           "SNI-D": r"SNI \Dm{}"}


def build(out_path: Path, summary_path: Path = T42) -> Path:
    d = json.loads(summary_path.read_text())
    counts = {(r["object"], r["condition"]): r
              for r in d["counts"] if r["kind"] == "inj"}
    for obj in OBJECTS:
        missing = [c for c in CONDS if (obj, c) not in counts]
        if missing:
            raise ValueError(f"{obj}: conditions absent from the summary: "
                             f"{missing} (all-or-refuse)")
    null_rate = d["null_detection_rate"]
    alpha = float(d["alpha"])

    rows = []
    for obj in OBJECTS:
        row = {"obj": DISPLAY.get(obj, obj)}
        for c, h in zip(CONDS, HEADS):
            r = counts[(obj, c)]
            row[c] = f"{r['detected']}/{r['n']}"
        row["null"] = f"{null_rate[obj]:.3f}"
        rows.append(row)
    body = pd.DataFrame(rows)

    above = sorted((o for o in OBJECTS if float(null_rate[o]) > alpha),
                   key=lambda o: -float(null_rate[o]))
    above_txt = (", ".join(
        rf"{DISPLAY.get(o, o)} ({float(null_rate[o]):.3f})" for o in above)
        if above else "none")
    style = TableStyle(environment="table*", notes=(
        INFO_ASYMMETRY_LEAKAGE + " " +
        rf"Detection counts out of 6 injections (2 datasets $\times$ 3 host "
        rf"seeds). Thresholds are calibrated on $\ge 20$ random-proxy "
        rf"retrainings per (object, target row, dataset); observed null "
        rf"rates are reported per object and batch (final column: this "
        rf"batch). "
        rf"Every count is a full host retraining, not a simulation. The "
        rf"discrepancy-control column reads under both estimands: the "
        rf"control is marginally correlated with the target but carries "
        rf"no incremental information, and a detection there therefore "
        rf"means different things under a dataset-risk estimand (where a "
        rf"marginal flag is expected behavior) and under a host-use "
        rf"fidelity estimand (where it is discordant with the "
        rf"zero-increment reference). A finite-sample host is not "
        rf"guaranteed to ignore the control: the same-host permutation "
        rf"reference itself indicated detectable use in 1 of 6 "
        rf"conditions, which is why these counts are reported as counts "
        rf"under two readings rather than as errors under one. "
        rf"The final "
        rf"column is the OBSERVED detection rate under the permutation "
        rf"null (shuffled proxies, n=42 per object), reported as a "
        rf"calibration diagnostic and not tested against the nominal "
        rf"$\alpha={alpha:g}$: that level is itself estimated from a "
        rf"finite random-proxy calibration sample. Objects whose observed "
        rf"rate sits above it: {above_txt}. Classes are never pooled. "
        rf"Layer naming: the detection rules and thresholds are the initial "
        rf"confirmatory layer (committed before the injection measurements); "
        rf"the frozen-generation confirmatory batch is a prospectively "
        rf"specified replication (rules committed before its measurements, "
        rf"after the original results were known). "
        rf"Construction generations and per-generation counts: Online "
        rf"Resource~1, leakage full readouts.",))
    tex = dataframe_to_tex(
        body,
        caption=(r"Leakage-risk discrimination (fourth \VERA{} axis): proxy-injection "
                 r"detection counts per object and condition, with the "
                 r"discrepancy control and permutation-null rates."),
        label="tab:leakage",
        column_format="lcccccccc",
        header=["Audit object"] + HEADS + ["Null rate"],
        style=style, escape_data=False)
    # Chat ruling 2026-08-29 (P5R SS4 writing obligation): the same-host
    # probe's interaction detections are never stated without its
    # P5R-K SS2 (fourth review SS4): the fixed-alpha exact binomial and the
    # (6/42)^6 chance bound are withdrawn. Thresholds are empirical 95th
    # percentiles of a finite calibration sample, so a tail probability
    # against a KNOWN 0.05 misstates what is known; the observed null rates
    # stay, as calibration diagnostics.
    try:
        pb = d["probe_interaction_chance_bound"]
    except KeyError as exc:
        raise ValueError(f"summary lacks {exc} -- rerun t42 analyze "
                         f"(all-or-refuse)") from exc
    macros = (
        "% generated by reporting/table_leakage.py -- probe null-rate macros\n"
        "% (P5R-K SS2: interaction detections are stated together with the\n"
        "% probe's observed null rate, reported as a calibration diagnostic;\n"
        "% no binomial tail against a fixed 0.05 and no chance bound)\n"
        f"\\newcommand{{\\probeNullFprCounts}}{{{pb['fpr_counts']}}}\n"
        f"\\newcommand{{\\probeNullFprRate}}"
        f"{{{pb['fpr_implemented']:.3f}}}\n"
        f"\\newcommand{{\\probeInteractionCounts}}"
        f"{{{pb['interaction_detected']}/{pb['interaction_n']}}}\n")
    (out_path.parent / "leak_macros.tex").write_text(macros)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_leakage.py",
        "input": str(summary_path),
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    def fake_summary(drop=None):
        counts = []
        for obj in OBJECTS:
            for c in CONDS:
                if drop and (obj, c) == drop:
                    continue
                counts.append({"object": obj, "condition": c, "kind": "inj",
                               "detected": 6 if obj == "P" else
                               (0 if (obj, c) == ("SNI-D", "interaction")
                                else 3),
                               "n": 6, "wilson_lo": 0, "wilson_hi": 1})
        return {"alpha": 0.05, "counts": counts,
                "null_detection_rate": {o: (0.143 if o == "Permutation-on-SNI"
                                            else 0.095 if o == "SNI-D"
                                            else 0.0) for o in OBJECTS},
                "null_exact_binomial": {"Permutation-on-SNI": {
                    "detected": 6, "n": 42,
                    "p_geq_k_exact_binomial": 0.017432}},
                "probe_interaction_chance_bound": {
                    "object": "Permutation-on-SNI",
                    "fpr_implemented": round(6 / 42, 6),
                    "fpr_counts": "6/42",
                    "interaction_detected": 6, "interaction_n": 6,
                    "p_all_detected_at_implemented_fpr": (6 / 42) ** 6}}

    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"
        sp.write_text(json.dumps(fake_summary()))
        out = build(Path(td) / "t.tex", sp)
        txt = out.read_text()
        check("0/6" in txt and txt.count("6/6") >= 7,
              "cells carry k/n verbatim (SNI-D interaction 0/6 present)")
        check("Permutation-on-SNI (0.143), SNI \\Dm{} (0.095)" in txt,
              "above-nominal objects named by computation, sorted desc")
        check("0.143" in txt and "0.095" in txt,
              "null-rate column prints both flagged rates")
        mac = (Path(td) / "leak_macros.tex").read_text()
        check("ChanceBound" not in mac and "BinomP" not in mac,
              "P5R-K SS2: the fixed-alpha binomial p and the chance bound "
              "are no longer emitted")
        check("{6/42}" in mac and "{0.143}" in mac and "{6/6}" in mac,
              "null counts/rate and interaction counts still emitted as "
              "calibration diagnostics")
        sp2 = Path(td) / "s_old.json"
        old = fake_summary()
        del old["probe_interaction_chance_bound"]
        sp2.write_text(json.dumps(old))
        try:
            build(Path(td) / "t_old.tex", sp2)
            check(False, "pre-ruling summary must refuse")
        except ValueError as e:
            check("rerun t42 analyze" in str(e),
                  "summary without the bound field refused")
        sp.write_text(json.dumps(fake_summary(drop=("SNI-D", condition_order()[-1]))))
        try:
            build(Path(td) / "t2.tex", sp)
            check(False, "missing condition must refuse")
        except ValueError as e:
            check("all-or-refuse" in str(e), "missing condition refused")
        # no objects above nominal -> the note says 'none'
        s = fake_summary()
        s["null_detection_rate"] = {o: 0.0 for o in OBJECTS}
        sp.write_text(json.dumps(s))
        out3 = build(Path(td) / "t3.tex", sp)
        check("above it: none" in out3.read_text(),
              "no-above-nominal branch prints 'none'")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_leakage.tex"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
