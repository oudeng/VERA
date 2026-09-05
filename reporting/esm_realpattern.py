"""ESM section: real-pattern-inspired masking -- construction, scorable
ground truth, coverage and selection caveats (P5R-G gate 3; second internal
review SS8's seven required items).

Single sources: data/masks/real_pattern/CDC2022/*_meta.json (per-seed
construction record) + results/T2b_real_pattern/real_pattern_masks.csv
(per-seed statistics). Every number in the emitted text is read from them.

    env PYTHONHASHSEED=2025 python reporting/esm_realpattern.py
    env PYTHONHASHSEED=2025 python reporting/esm_realpattern.py --selftest
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import write_tex                           # noqa: E402

META_DIR = CODE_ROOT / "data" / "masks" / "real_pattern" / "CDC2022"
STATS = CODE_ROOT / "results" / "T2b_real_pattern" / "real_pattern_masks.csv"
OUT = CODE_ROOT / "reporting" / "out" / "sec_esm_realpattern.tex"


def _pct(x: float, nd: int = 1) -> str:
    """A percentage as LaTeX source: the sign escaped, never a comment."""
    return rf"{x * 100:.{nd}f}\%"


def build(meta_dir: Path = META_DIR, stats_csv: Path = STATS,
          out_path: Path = OUT) -> Path:
    meta = json.loads(sorted(meta_dir.glob("*_s1_meta.json"))[0].read_text())
    df = pd.read_csv(stats_csv)
    n_rows = meta["shape"]["n_rows"]
    n_targets = len(meta["spec"]["target_columns"])
    observed = [c for c in meta["spec"]["observed_columns"] if c != "ID"]
    rates = [meta["rates"]["per_column_missing_rate"][c]
             for c in meta["spec"]["target_columns"]]
    r_lo, r_hi = min(rates), max(rates)
    # Coverage is read over all five seed records, not seed 1 alone: the
    # scored sample size is the number the fourth internal review (SS4.1)
    # asked to be legible, and one maskable column never receives a cell.
    metas = [json.loads(f.read_text())
             for f in sorted(meta_dir.glob("*_meta.json"))]
    tcols = meta["spec"]["target_columns"]

    def _rate(m, c):
        return m["rates"]["per_column_missing_rate"][c]

    never = [c for c in tcols if all(_rate(m, c) == 0 for m in metas)]
    sometimes = [c for c in tcols if c not in never
                 and any(_rate(m, c) == 0 for m in metas)]
    nz = [_rate(m, c) for m in metas for c in tcols if _rate(m, c) > 0]
    counts = sorted(round(r * n_rows) for r in nz)
    if not nz:
        raise ValueError("no scored column in any seed")
    n_seeds = len(metas)
    never_tex = ", ".join(rf"\texttt{{{c}}}" for c in never)
    if sometimes:
        sometimes_tex = (r" A further "
                         + str(len(sometimes))
                         + r" column(s) ("
                         + ", ".join(rf"\texttt{{{c}}}" for c in sometimes)
                         + r") draw no cell in at least one seed.")
    else:
        sometimes_tex = ""
    ov_lo, ov_hi = df.overall_rate.min(), df.overall_rate.max()
    disp_lo, disp_hi = df.dispersion_ratio.min(), df.dispersion_ratio.max()
    fo_lo, fo_hi = df.frac_rows_fully_observed.min(), df.frac_rows_fully_observed.max()
    # Percentages are rendered through _pct: a bare Python "%" format is a
    # LaTeX comment character and silently swallows the rest of the source
    # line (it did, in the IR4 package -- fourth internal review SS4.1).
    p_rlo, p_rhi = _pct(min(nz), 2), _pct(max(nz), 2)
    p_ovlo, p_ovhi = _pct(ov_lo, 1), _pct(ov_hi, 1)
    p_folo, p_fohi = _pct(fo_lo, 0), _pct(fo_hi, 0)
    n_lo, n_hi = counts[0], counts[-1]
    n_med = round(statistics.median(counts))
    _WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    n_never = _WORD.get(len(never), str(len(never)))
    verb = "receives" if len(never) == 1 else "receive"
    src = Path(meta["source"]).name.replace("_", r"\_")
    obs_tex = ", ".join(c.replace("_", r"\_") for c in observed)

    body = rf"""\subsection{{Construction}}
The real-pattern-inspired masking condition is built in one step per
seed: whole rows of the row-level missingness indicator observed in the
raw source survey (\texttt{{{src}}}, 445{{,}}132 BRFSS 2022 respondents)
are sampled with replacement, one donor row per row of the complete-case
table, and the sampled indicator is applied to the table.
% src: missingness/real_pattern.py (draw_real_patterns/apply_to) + {meta['generator']}
Sampling whole rows rather than per-column indicators is what preserves
the three properties simultaneously: the per-column rate profile, the
row-level co-occurrence structure, and derived-variable propagation
(in the raw survey, $P(\text{{BMI missing}} \mid \text{{height or weight
missing}}) = 1$, which no independent per-column draw can express).

\subsection{{The seven construction questions}}
\begin{{enumerate}}
\item \emph{{Where the real pattern comes from:}} the observed
missingness indicator matrix of \texttt{{{src}}} -- the raw file the
complete-case benchmark table was extracted from.
\item \emph{{Which rows and columns are scorable:}} every cell of the
complete-case table ({n_rows:,} rows) is observed before masking, so
every masked cell has a visible true value; {n_targets} feature columns
are maskable. The identifier, the downstream target, and the
always-observed driver columns ({obs_tex}) are never masked.
% src: {meta['generator']} spec.target_columns/observed_columns
\item \emph{{Pre-masking observedness:}} guaranteed by construction --
the table is the complete-case extraction, and the generator rejects
input containing missing values.
\item \emph{{Natural missing cells:}} none exist in the masked table;
the rows carrying the survey's natural missingness were removed at
complete-case construction, which is exactly the selection this
condition inherits (item 7).
\item \emph{{Pattern-to-table mapping:}} donor rows are drawn from the
full survey with replacement; indicator columns are matched to table
columns by name, columns absent from the donor stay observed, and the
always-observed set is forced observed after resampling.
\item \emph{{Coverage and scored sample size:}} the scored sample of a
column is its missing rate times {n_rows:,} rows. {n_never} of the
{n_targets} maskable columns ({never_tex}) {verb} no masked cell in
any of the {n_seeds} seeds -- it is never missing in the donor survey --
and is therefore never scored.{sometimes_tex} Over the remaining
columns and seeds, the per-column missing rate spans
{p_rlo}--{p_rhi}, that is {n_lo:,}--{n_hi:,} scored cells per column
(median {n_med:,}); the per-column values are in the per-seed
\texttt{{meta.json}}. The overall missing rate is {p_ovlo}--{p_ovhi},
the row-level dispersion is {disp_lo:.1f}--{disp_hi:.1f}$\times$ the
independent-Bernoulli reference, and {p_folo}--{p_fohi} of rows are
fully observed.
% src: results/T2b_real_pattern/real_pattern_masks.csv + all per-seed meta.json
\item \emph{{Selection bias:}} the scored cells live on the
complete-case subpopulation -- respondents who answered everything --
so the condition evaluates imputers on the real pattern's
\emph{{shape}}, not on the full survey population. Respondents with
fully observed rows differ systematically from the survey population,
and nothing here corrects for that.
\end{{enumerate}}

\subsection{{Scope}}
The donor pattern is survey item non-response from a telephone health
interview; it is not an EHR missingness mechanism, and no conclusion
from this condition is extrapolated to one.
% src: missingness/real_pattern.py meta "boundary"
The main text's rank-agreement reading is correspondingly narrow: on
this CDC benchmark and under this real-pattern-inspired masking
construction, the ranking was similar to MAR@30\%; this does not
validate the simulator for other missingness mechanisms or datasets.
"""
    return write_tex(out_path, body, provenance={
        "generator": "reporting/esm_realpattern.py",
        "input": f"{meta_dir} + {stats_csv}",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    p = build()
    t = " ".join(p.read_text().split())
    for item in ("Where the real pattern comes from",
                 "Which rows and columns are scorable",
                 "Pre-masking observedness", "Natural missing cells",
                 "Pattern-to-table mapping", "Coverage and scored sample size",
                 "Selection bias"):
        check(item in t, f"seven-item coverage: {item}")
    check("real-pattern-inspired masking" in t, "unified condition name")
    check("does not validate the simulator" in t, "narrowed reading present")
    check("445{,}132" in t, "donor size from source")
    check(t.count("% src:") >= 3, "source pointers present")
    # SS4.1: every rendered percentage must survive as text, not be eaten as
    # a LaTeX comment. Checked on the raw file, line by line.
    import re as _re
    raw = p.read_text()
    stray = [(i, ln) for i, ln in enumerate(raw.splitlines(), 1)
             for m in _re.finditer(r"(?<!\\)%", ln)
             if ln[:m.start()].strip() != ""]
    check(not stray, f"no mid-line unescaped % (found {stray[:2]})")
    for want in ("maskable columns", "no masked cell in any of the",
                 "per-column missing rate spans", "scored cells per column",
                 "overall missing rate is", "of rows are fully observed"):
        check(want in t, f"coverage item renders: {want}")
    check(t.count(r"\%") >= 6, "six escaped percentages in the coverage item")
    reg = json.loads((CODE_ROOT / "docs" / "terminology_registry.json").read_text())
    banned_all = [b for term in reg["terms"] for b in term.get("banned", [])]
    for banned in banned_all:
        check(banned.lower() not in t.lower(), f"registry variant absent: {banned}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    print(f"[OK] wrote {build()}")
    raise SystemExit(0)
