"""Real-vs-simulated missingness table (P5 SS3.1): CDC2022 under the
grid's synthetic MAR@30 masks against the real-missingness condition
(REAL_PATTERN), per method, medians over the five seeds, with per-condition
ranks. Everything is derived from the grid artifacts.

    PYTHONHASHSEED=2025 python reporting/table_realgap.py [--selftest]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import display_name, TableStyle, dataframe_to_tex, write_tex  # noqa: E402

GRID = CODE_ROOT / "results" / "P2_main_grid"
METHODS = ["MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE",
           "HyperImpute", "TabCSDI", "SNI"]


def load(grid: Path = GRID) -> pd.DataFrame:
    from stats.aggregate_grid import load_runs
    df = load_runs(grid)
    return df[df.dataset == "CDC2022"]


def build(out_path: Path, grid: Path = GRID) -> Path:
    df = load(grid)
    synth = df[(df.mechanism == "MAR") & (df.rate == 0.3)]
    rp = df[df.mechanism == "REAL_PATTERN"]
    rows = []
    for cond_name, sel in (("synthetic", synth), ("real", rp)):
        for m in METHODS:
            g = sel[sel.method == m]
            if len(g) != 5:
                raise ValueError(f"{cond_name}/{m}: {len(g)} seeds, need 5")
            rows.append({"cond": cond_name, "method": m,
                         "nrmse": float(g.cont_NRMSE.median()),
                         "f1": float(g["cat_Macro-F1"].median())})
    t = pd.DataFrame(rows)
    piv_n = t.pivot(index="method", columns="cond", values="nrmse")
    piv_f = t.pivot(index="method", columns="cond", values="f1")
    rank_syn = piv_n["synthetic"].rank()
    rank_real = piv_n["real"].rank()

    body = pd.DataFrame({
        "Method": [display_name(m) for m in METHODS],
        "syn_n": [f"{piv_n.loc[m, 'synthetic']:.3f}" for m in METHODS],
        "real_n": [f"{piv_n.loc[m, 'real']:.3f}" for m in METHODS],
        "syn_f": [f"{piv_f.loc[m, 'synthetic']:.3f}" for m in METHODS],
        "real_f": [f"{piv_f.loc[m, 'real']:.3f}" for m in METHODS],
        "rank": [f"{int(rank_syn[m])}$\\to${int(rank_real[m])}"
                 for m in METHODS]})

    from scipy.stats import spearmanr
    rho = float(spearmanr(rank_syn, rank_real).statistic)
    sni_shift = f"{int(rank_syn['SNI'])}$\\to${int(rank_real['SNI'])}"
    style = TableStyle(environment="table*", notes=(
        rf"Medians over five seeds on CDC2022; the synthetic column is the "
        rf"grid's MAR@30\% condition, the real-pattern column applies "
        rf"real-pattern-inspired masking: whole rows of the missingness "
        rf"indicator observed in the raw source survey, resampled onto the "
        rf"complete-case table so every masked cell has a scorable true "
        rf"value (construction, per-column coverage and selection caveats: "
        rf"Online Resource~1). Rank is by NRMSE "
        rf"(1 = best) under each condition; the two orderings correlate at "
        rf"Spearman $\rho={rho:.2f}$, and SNI moves {sni_shift}. This "
        rf"agreement is specific to this table and masking construction; "
        rf"it does not validate the simulator for other missingness "
        rf"mechanisms or datasets. "
        rf"Every value is a grid artifact; no re-runs.",))
    tex = dataframe_to_tex(
        body, caption=(r"Real versus simulated missingness on CDC2022: the "
                       r"same nine imputers under the synthetic MAR mask "
                       r"and under real-pattern-inspired masking."),
        label="tab:realgap", column_format="lccccc",
        header=["Method", r"NRMSE (syn)", r"NRMSE (real)",
                r"Macro-F1 (syn)", r"Macro-F1 (real)", r"Rank syn$\to$real"],
        style=style, escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_realgap.py",
        "input": str(grid) + " (CDC2022 MAR@30 + REAL_PATTERN cells)",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    import json
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        g = Path(td)
        for mech, rate in (("MAR", 0.3), ("REAL_PATTERN", None)):
            for mi, m in enumerate(METHODS):
                for s in (1, 2, 3, 5, 8):
                    tag = (f"CDC2022_{mech}_30per_{m}_s{s}" if rate
                           else f"CDC2022_{mech}_{m}_s{s}")
                    d = g / tag
                    d.mkdir()
                    (d / "metrics_summary.json").write_text(json.dumps({
                        "dataset": "CDC2022", "mechanism": mech,
                        "rate": rate, "method": display_name(m), "seed": s,
                        "cont_NRMSE": 0.1 * (mi + 1) + (0.5 if mech ==
                                                        "REAL_PATTERN" and
                                                        m == "SNI" else 0),
                        "cat_Macro-F1": 0.5}))
        out = build(g / "t.tex", grid=g)
        txt = out.read_text()
        check("$\\to$" in txt, "rank-shift column present")
        # crafted: SNI (index 8, best-9th... nrmse 0.9 syn; +0.5 real -> 1.4
        # stays rank 9 both) -> shift 9->9; MeanMode 0.1 -> rank 1 both
        check("1$\\to$1" in txt and "9$\\to$9" in txt,
              "crafted ranks land exactly (1->1 best, 9->9 SNI)")
        check("\\rho=1.00" in txt,
              "identical orderings -> Spearman rho = 1.00 in the note")
        # missing seed -> refusal
        victim = next(g.glob("CDC2022_REAL_PATTERN_SNI_s8"))
        (victim / "metrics_summary.json").unlink()
        try:
            build(g / "t2.tex", grid=g)
            check(False, "missing seed must refuse")
        except ValueError as e:
            check("need 5" in str(e), "missing seed refused")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_realgap.tex"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
