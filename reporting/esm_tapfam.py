"""ESM table: the TAP baseline family (P5R-H SS7.1; rules 89c386d).

Single source: results/T5_family/tapfam_summary.json. Prospectively
specified sensitivity layer -- contextual placement only, no verdict.

    env PYTHONHASHSEED=2025 python reporting/esm_tapfam.py [--selftest]
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
from reporting.latex import signed, TableStyle, dataframe_to_tex, write_tex  # noqa: E402

SRC = CODE_ROOT / "results" / "T5_family" / "tapfam_summary.json"
OUT = CODE_ROOT / "reporting" / "out" / "tab_esm_tapfam.tex"
DISPLAY = {"abs_spearman": r"$|$Spearman$|$ (type-aware)",
           "mutual_information": "Normalized MI (8-bin)",
           "observed_only_tap": "Observed-only TAP (pairwise)"}


def build(src: Path = SRC, out_path: Path = OUT) -> Path:
    d = json.loads(src.read_text())
    rows = []
    for ds, blk in d["datasets"].items():
        for v in ("abs_spearman", "mutual_information", "observed_only_tap"):
            f = blk[v]
            rows.append({"Table": ds, "Variant": DISPLAY[v],
                         r"$T$ vs \TAP{}": signed(f['T']),
                         r"95\% CI": (f"[{signed(f['ci95_T'][0])}, "
                                      f"{signed(f['ci95_T'][1])}]"),
                         r"exact $p$": f"{f['p_exact']:.3f}",
                         "n seeds": f["n_seeds"]})
        r = blk["random"]
        rows.append({"Table": ds, "Variant":
                     f"Random (floor band, {r['n_replicates']} replicates)",
                     r"$T$ vs \TAP{}": (f"{signed(r['T_min'])} / "
                                        f"{signed(r['T_median'])} / "
                                        f"{signed(r['T_max'])}"),
                     r"95\% CI": "min / median / max",
                     r"exact $p$": "--", "n seeds": blk["n_seeds_found"]})
        rows.append({"Table": ds, "Variant": "Uniform (constant)",
                     r"$T$ vs \TAP{}": "degenerate",
                     r"95\% CI": "no ranking information",
                     r"exact $p$": "--", "n seeds": "--"})
    body = pd.DataFrame(rows)
    # The band is quoted from the artifact, not typed: the correction of
    # 2026-08-29 moved these numbers, and an adjective ("narrow") that does
    # not move with them would go quietly out of date.
    assoc = [abs(float(d["datasets"][ds][v]["T"]))
             for ds in d["datasets"]
             for v in ("abs_spearman", "mutual_information",
                       "observed_only_tap")]
    rand = [float(d["datasets"][ds]["random"][k])
            for ds in d["datasets"] for k in ("T_min", "T_max")]
    band = (rf"every association-summary variant lands within "
            rf"$|T| \le {max(assoc):.3f}$ of \TAP{{}}$_0$, while the random "
            rf"replicates span ${min(rand):+.3f}$ to ${max(rand):+.3f}$")
    tex = dataframe_to_tex(
        body, caption=(r"The training-free baseline family: faithfulness-"
                       r"axis placement of each variant relative to "
                       r"\TAP{}$_0$ (positive $T$ = variant closer to the "
                       r"behavioral reference)."),
        label="tab:esm_tapfam", column_format="llcccc",
        header=list(body.columns),
        style=TableStyle(environment="table*", notes=(
            r"Prospectively specified sensitivity analysis (rules "
            r"committed before any family readout, \texttt{89c386d}); "
            r"contextual placement only -- no committed verdict reads "
            r"from this table. $T$ = mean of seed-level median "
            r"$\Delta\rho$(variant $-$ \TAP{}) under the unified "
            r"estimand; every variant is computed on frozen inputs with "
            r"zero retraining (the archived ablation matrices are the "
            r"reference). Two variants -- abs-Spearman and normalized "
            r"MI -- were recomputed on 2026-08-29 after a lineage audit "
            r"found they had been computed on the pre-mask table rather "
            r"than on the initial completion the archived \TAP{}$_0$ used; "
            r"the superseded readouts and the difference are recorded in "
            r"the correction record. On the corrected inputs, " + band +
            r". This places \TAP{} within that comparator family, but "
            r"it does not by itself explain \Dm{}'s proximity to \TAP{}: "
            r"\Dm{} is not an association summary, and the family result "
            r"bounds how much of the placement chance could account for, "
            r"not why the attention readout sits there.",)),
        escape_data=False)
    tex = tex.replace(r"\scriptsize",
                      r"\scriptsize\setlength{\tabcolsep}{3pt}", 1)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/esm_tapfam.py", "input": str(src),
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    fam = {"datasets": {"X": {
        "n_seeds_found": 15,
        "abs_spearman": {"T": -0.026, "ci95_T": [-0.05, -0.002],
                         "p_exact": 0.067, "n_seeds": 15},
        "mutual_information": {"T": -0.001, "ci95_T": [-0.03, 0.02],
                               "p_exact": 0.94, "n_seeds": 15},
        "observed_only_tap": {"T": 0.007, "ci95_T": [-0.005, 0.019],
                              "p_exact": 0.257, "n_seeds": 15},
        "random": {"n_replicates": 20, "T_min": -0.379,
                   "T_median": -0.243, "T_max": -0.176},
        "uniform": {"degenerate": True}}}}
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"
        sp.write_text(json.dumps(fam))
        out = build(sp, Path(td) / "t.tex")
        txt = out.read_text()
        check("-0.026" in txt and "0.067" in txt, "T and p carried")
        check("degenerate" in txt, "uniform degeneracy stated")
        check("-0.379" in txt and "-0.176" in txt, "random floor band")
        check("no committed verdict" in " ".join(txt.split()),
              "sensitivity-layer note present")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    print(f"[OK] wrote {build()}")
    raise SystemExit(0)
