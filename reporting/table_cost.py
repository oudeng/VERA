"""R4.6 -- cost table under BOTH accountings (Results, cost axis).

Input: results/T3_five_way/fiveway_cost.csv. Guard (P4-D): the marginal
accounting (audit seconds given the imputer already ran) and the total
accounting (everything paid to obtain the artifact) MUST both appear, or the
generator refuses. R0 published only the accounting that favored the
attention matrix; this guard exists so we cannot repeat that, even by
accident.

    PYTHONHASHSEED=2025 python reporting/table_cost.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
T42 = CODE_ROOT / "results" / "T4_leakage"
DATASETS = ["MIMIC", "eICU"]
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest", "Permutation-on-SNI"]
DISPLAY = {"P": r"\TAP{}", "MissForest-importance": "MF importance",
           "SHAP-on-MissForest": "SHAP-on-MF",
           "Permutation-on-MissForest": "Permutation-on-MF",
           "Permutation-on-SNI": "Permutation-on-SNI",
           "SNI-D": r"SNI \Dm{}"}


def _fmt(sec: float) -> str:
    return f"{sec:,.1f}" if sec < 100 else f"{sec:,.0f}"


def _aux_windows(t42_dir: Path = T42, registry: Path | None = None) -> list:
    """Auxiliary-load windows to annotate against (P4L SS5): the entries of
    the registry file (results/aux_windows.json) plus, if T4.2 worker
    window files exist, one derived envelope entry. Each window is
    {label, start_unix, end_unix (None = still open), nice}. Empty list =
    no auxiliary load ever registered -> no sentence."""
    import json
    registry = registry or (CODE_ROOT / "results" / "aux_windows.json")
    wins = []
    if registry.exists():
        for w in json.loads(registry.read_text()):
            wins.append({"label": w["label"],
                         "start_unix": float(w["start_unix"]),
                         "end_unix": (None if w.get("end_unix") is None
                                      else float(w["end_unix"])),
                         "nice": int(w.get("nice", 10))})
    recs = [json.loads(p.read_text())
            for p in sorted(t42_dir.glob("window_*.json"))]
    if recs:
        wins.append({"label": "leakage study",
                     "start_unix": min(r["start_unix"] for r in recs),
                     "end_unix": max(r["end_unix"] for r in recs),
                     "nice": 10})
    return wins


def _overlap_by_window(intervals: dict, windows: list) -> dict:
    """Pure intersection: {label: sorted cell tags whose [start, end)
    intersects the window}. An open window (end None) extends to +inf --
    the conservative direction: it can only flag more cells, never fewer."""
    out = {}
    for w in windows:
        w0 = w["start_unix"]
        w1 = w["end_unix"] if w["end_unix"] is not None else float("inf")
        out[w["label"]] = sorted(t for t, (s, e) in intervals.items()
                                 if s < w1 and e > w0)
    return out


def _window_note(windows: list, hits: dict, n_cells: int) -> str:
    """Transparency sentences, one per window; empty string when no window
    is registered. Numbers in the table are never changed by this."""
    import time as _time
    if not windows:
        return ""
    parts = []
    for w in windows:
        k = len(hits[w["label"]])
        t0 = _time.strftime("%b %d %H:%M", _time.localtime(w["start_unix"]))
        t1 = ("end pending" if w["end_unix"] is None else
              _time.strftime("%b %d %H:%M", _time.localtime(w["end_unix"])))
        parts.append(
            rf"Of the {n_cells} fit-source cells, {k} ran wholly or partly "
            rf"inside the {w['label']} window ({t0}--{t1}; auxiliary load "
            rf"at nice {w['nice']}, so the grid held scheduling priority). ")
    return ("".join(parts)
            + r"Affected cells are listed in the generated file's "
              r"provenance comment; no number is changed by these "
              r"annotations. ")


def _grid_fits(grid_root: Path, windows: list) -> tuple:
    """Fit column from the grid's own wall clocks (C-6 target state).

    Guard: MIMIC/eICU x {SNI, MissForest} x MAR@30 x 5 seeds -- all 20 cells
    or refusal. These are the same records the ESM runtime table draws from,
    which is what makes the two tables one accounting instead of two.

    Also annotates (P4K SS1.3 / P4L SS5): which of the 20 source cells ran
    wholly or partly inside each registered auxiliary-load window. A cell's
    interval is [mtime(metrics_summary.json) - runtime_sec, mtime].
    """
    from stats.aggregate_grid import load_runs
    runs = load_runs(grid_root)
    g = runs[(runs.mechanism == "MAR") & (runs.rate == 0.3)
             & runs.dataset.isin(DATASETS)
             & runs.method.isin(["SNI", "MissForest"])]
    fits = {}
    intervals = {}
    for ds in DATASETS:
        for meth in ["SNI", "MissForest"]:
            sel = g[(g.dataset == ds) & (g.method == meth)]
            v = pd.to_numeric(sel.runtime_sec, errors="coerce").dropna()
            if len(v) < 5:
                raise ValueError(
                    f"--fit-from-grid: {ds}x{meth} MAR@30 has {len(v)}/5 "
                    f"grid cells; refusing a mixed-source fit column "
                    f"(C-6).")
            fits[(ds, meth)] = float(v.median())
            for _, r in sel.iterrows():
                tag = f"{ds}_MAR_30per_{meth}_s{int(r.seed)}"
                ms = grid_root / tag / "metrics_summary.json"
                if not ms.exists():
                    raise ValueError(f"overlap annotation: {ms} missing "
                                     f"though load_runs returned the row")
                end = ms.stat().st_mtime
                intervals[tag] = (end - float(r.runtime_sec), end)
    return fits, _overlap_by_window(intervals, windows)


def build(out_path: Path, fit_from_grid: Path | None = None,
          t42_win_dir: Path = T42, aux_registry: Path | None = None) -> Path:
    cost = pd.read_csv(FIVEWAY / "fiveway_cost.csv")
    for col in ("impute_sec", "audit_sec"):     # the dual-accounting guard
        if col not in cost.columns:
            raise ValueError(
                f"cost data lacks '{col}': one accounting alone cannot be "
                f"published. R0 reported only the marginal reading; this "
                f"table exists to carry both, or nothing.")
    pcost = pd.read_csv(T4F / "perm_on_sni_audit_cost.csv")
    for ds in DATASETS:
        have = set(cost[cost.dataset == ds].method.unique())
        have |= ({"Permutation-on-SNI"}
                 if len(pcost[pcost.dataset == ds]) else set())
        missing = [o for o in OBJECTS if o not in have]
        if missing:
            raise ValueError(f"{ds}: objects missing from cost data: {missing} "
                             f"(six-or-refuse, docs/T4F_presentation_rule.md)")

    med = (cost.groupby(["dataset", "method"])[["impute_sec", "audit_sec"]]
           .median())
    windows, hits = [], {}
    if fit_from_grid is not None:
        windows = _aux_windows(t42_win_dir, aux_registry)
        fits, hits = _grid_fits(fit_from_grid, windows)
        for ds in DATASETS:
            med.loc[(ds, "SNI-D"), "impute_sec"] = fits[(ds, "SNI")]
            for m in ("MissForest-importance", "SHAP-on-MissForest",
                      "Permutation-on-MissForest"):
                med.loc[(ds, m), "impute_sec"] = fits[(ds, "MissForest")]
    rows = []
    totals: dict = {}
    for obj in OBJECTS:
        row = {"obj": DISPLAY.get(obj, obj)}
        for ds in DATASETS:
            if obj == "Permutation-on-SNI":
                i = float(med.loc[(ds, "SNI-D"), "impute_sec"])
                a = float(pcost[pcost.dataset == ds].audit_sec.iloc[0])
            else:
                i, a = med.loc[(ds, obj)]
            row[f"{ds}_i"] = _fmt(i)
            row[f"{ds}_a"] = _fmt(a)
            row[f"{ds}_t"] = _fmt(i + a)
            totals[(ds, obj)] = float(i) + float(a)
        rows.append(row)
    body = pd.DataFrame(rows)

    # P4O SS4.3: the prose ratio ("SNI-D at N x the total cost") is emitted
    # as macros from the SAME numbers as this table, never hand-copied.
    # Definition, verbatim from the ratio's birth record (P3_T31_report SS2:
    # "SNI-D vs 'MF + the most expensive explainer'"): per dataset,
    # total(SNI-D) / max over the MF-family readouts of total(readout);
    # the range is over the two datasets. Interim-sourced values were
    # 5540/248 ~ 22x (MIMIC) and 3733/50 ~ 75x (eICU).
    mf_objs = ["MissForest-importance", "SHAP-on-MissForest",
               "Permutation-on-MissForest"]
    per_ds = {ds: totals[(ds, "SNI-D")]
              / max(totals[(ds, m)] for m in mf_objs)
              for ds in DATASETS}
    lo, hi = min(per_ds.values()), max(per_ds.values())
    macros = (
        "% generated by reporting/table_cost.py -- prose cost-ratio macros\n"
        "% definition (P3_T31 SS2 caliber): total(SNI-D) / total(most\n"
        "% expensive MF-family readout), per dataset; range over datasets.\n"
        + "".join(f"% {ds}: {v:.2f}x\n" for ds, v in per_ds.items())
        + f"\\newcommand{{\\costRatioMin}}{{{round(lo)}}}\n"
        f"\\newcommand{{\\costRatioMax}}{{{round(hi)}}}\n"
        f"\\newcommand{{\\costRatioRange}}"
        f"{{{round(lo)}--{round(hi)}$\\times$}}\n")
    (out_path.parent / "cost_macros.tex").write_text(macros)

    # A3 single-thread standardized macros (P5R-C SS3.5): same total
    # caliber (SNI total / most expensive MF readout path), re-measured
    # single-threaded on an idle machine, one object per process
    # (experiments/cost_probe.py). All-or-refuse: missing A3 records
    # would leave the prose macros undefined, so the build fails here.
    # The single-thread macros are emitted by
    # reporting/table_cost_primary.py, which owns the primary panel;
    # one owner per macro so the table and the prose cannot drift.


    if fit_from_grid is not None:
        cond = (r"Fit times are the full grid's recorded wall clocks "
                r"(12-way CPU queue, BLAS threads pinned at 2; the same "
                r"records as the runtime table in Online Resource~1, so the two tables share "
                r"one accounting -- the shared cost-window registry). ")
        cond += _window_note(windows, hits, 20)
    else:
        cond = (r"INTERIM SOURCING (the shared cost-window registry): fit times here were "
                r"measured in the stability study under 10 concurrent runs "
                r"and are inflated by that contention; they will be "
                r"re-sourced from the grid's wall clocks, the same records "
                r"as the runtime table in Online Resource~1, when the grid completes. Audit "
                r"times are unaffected. ")
    style = TableStyle(environment="table*", notes=(
        cond +
        r"\textit{Two accountings, both required:} the \emph{marginal} "
        r"reading (audit column) prices the artifact given that its host "
        r"imputer runs anyway -- the reading under which the attention "
        r"matrix's zero marginal cost is real; the \emph{total} reading "
        r"prices everything paid to obtain the artifact, host model "
        r"included -- the reading under which it is the most expensive "
        r"object in the comparison. Our earlier submission reported only "
        r"the former. Median seconds per run; \TAP{} needs no host model. "
        r"Wall-clock seconds are records of the stated implementation on "
        r"one machine with heterogeneous performance/efficiency cores "
        r"(i9-13900K): identical jobs ran up to $3.7\times$ faster alone "
        r"on a performance core than under a contended queue, so ratios, "
        r"not absolute seconds, are the comparable quantity on this "
        r"machine; neither absolute times nor ratios are assumed to "
        r"transfer across systems. This table is the operational log of "
        r"the campaign as executed (grid concurrency); the primary "
        r"single-thread idle-machine benchmark is reported in the text "
        r"from dedicated per-object probes, with per-object memory in "
        r"the Online Resource.",))
    tex = dataframe_to_tex(
        body,
        caption=(r"Operational cost log (grid concurrency) of each audit "
                 r"object under marginal and total "
                 r"accounting (median seconds per run)."),
        label="tab:cost",
        column_format="lrrrrrr",
        header=["Audit object", "MIMIC fit", "audit", "total",
                "eICU fit", "audit", "total"],
        style=style, escape_data=False)
    body_txt = tex
    assert "marginal" in body_txt and "total" in body_txt, \
        "dual-accounting wording missing from the emitted note"
    prov = {"generator": "reporting/table_cost.py",
            "input": str(FIVEWAY / "fiveway_cost.csv"),
            "code_SNI commit": runconfig.git_commit()}
    if fit_from_grid is not None:
        prov["input_fit_runtimes"] = (
            f"{fit_from_grid}/<tag>/metrics_summary.json runtime_sec "
            f"(MIMIC/eICU x SNI/MissForest x MAR@30 x 5 seeds)")
    for label, cells_ in hits.items():
        prov[f"aux_overlap[{label}]"] = ",".join(cells_) if cells_ else "none"
    return write_tex(out_path, tex, provenance=prov)


def _a3_ratios(a3_dir: Path):
    """Single-thread total-caliber ratios + peak RSS list from the A3
    probe records. Pure given the directory; raises on missing files."""
    import json as _json
    import re as _re
    st, rss = {}, []
    for ds in DATASETS:
        sni = _json.loads((a3_dir / f"{ds}_SNI.json").read_text())
        mf = _json.loads((a3_dir / f"{ds}_MF.json").read_text())
        worst = max(v["impute_sec"] + v["readout_sec"]
                    for v in mf["per_readout_sec"].values())
        st[ds] = float(sni["wall_total_sec"]) / worst
        for obj in ("P", "MF", "SNI"):
            t = (a3_dir / f"{ds}_{obj}_time.txt").read_text()
            m = _re.search(r"Maximum resident set size \(kbytes\): (\d+)",
                           t)
            rss.append(int(m.group(1)) / 1e6)
    return st, rss


def _selftest() -> int:
    """Known-answer checks for the auxiliary-window machinery (P4L SS5.3).
    The pure split (note builder returns text; the body is built before any
    window is consulted) is what guarantees numbers cannot be changed; the
    full-pipeline byte-identity is additionally verified in the shell on
    each refactor and recorded in the receipt."""
    import json
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    iv = {"c1": (100.0, 200.0), "c2": (300.0, 400.0), "c3": (150.0, 350.0)}
    W = [{"label": "W1", "start_unix": 120.0, "end_unix": 180.0, "nice": 10},
         {"label": "W2", "start_unix": 160.0, "end_unix": 360.0, "nice": 10}]
    hits = _overlap_by_window(iv, W)
    check(hits["W1"] == ["c1", "c3"], "overlapping windows: W1 hits c1,c3")
    check(hits["W2"] == ["c1", "c2", "c3"], "overlapping windows: W2 hits all")
    X = [{"label": "A", "start_unix": 0.0, "end_unix": 50.0, "nice": 10},
         {"label": "B", "start_unix": 500.0, "end_unix": 600.0, "nice": 10}]
    hx = _overlap_by_window(iv, X)
    check(hx["A"] == [] and hx["B"] == [],
          "mutually exclusive windows: both empty")
    O = [{"label": "open", "start_unix": 350.0, "end_unix": None, "nice": 10}]
    check(_overlap_by_window(iv, O)["open"] == ["c2"],
          "open window [350,inf): c2 in, c3 (ends exactly 350) out (strict)")
    check(_window_note([], {}, 20) == "",
          "no window registered -> no sentence at all")
    note = _window_note(W[:1], {"W1": ["c1", "c3"]}, 20)
    check("2 ran wholly or partly inside the W1 window" in note
          and "nice 10" in note, "note carries per-window count and nice")
    check("end pending" in _window_note(O, {"open": ["c2"]}, 20),
          "open window note says 'end pending'")
    with tempfile.TemporaryDirectory() as td:
        reg = Path(td) / "aux_windows.json"
        reg.write_text(json.dumps([{"label": "no-prior control study",
                                    "start_unix": 1.0, "end_unix": None,
                                    "nice": 10}]))
        wd = Path(td) / "t42"
        wd.mkdir()
        wins = _aux_windows(wd, reg)
        check([w["label"] for w in wins] == ["no-prior control study"]
              and wins[0]["end_unix"] is None,
              "registry loads; null end preserved; no T4.2 files -> no entry")
        (wd / "window_0.json").write_text(json.dumps(
            {"start_unix": 5.0, "end_unix": 9.0}))
        (wd / "window_1.json").write_text(json.dumps(
            {"start_unix": 4.0, "end_unix": 11.0}))
        wins = _aux_windows(wd, reg)
        t42w = [w for w in wins if w["label"] == "leakage study"]
        check(len(t42w) == 1 and t42w[0]["start_unix"] == 4.0
              and t42w[0]["end_unix"] == 11.0,
              "the leakage-study entry is derived as the envelope of the worker files")
    # A3 ratio helper on a hand-built fixture: SNI 200s vs worst MF path
    # 10+40=50s -> 4x; RSS 1e6 kB = 1.0 GB.
    with tempfile.TemporaryDirectory() as td:
        a3 = Path(td)
        for ds in DATASETS:
            (a3 / f"{ds}_SNI.json").write_text(json.dumps(
                {"wall_total_sec": 200.0}))
            (a3 / f"{ds}_MF.json").write_text(json.dumps(
                {"per_readout_sec": {
                    "A": {"impute_sec": 10.0, "readout_sec": 40.0},
                    "B": {"impute_sec": 10.0, "readout_sec": 5.0}}}))
            for obj in ("P", "MF", "SNI"):
                (a3 / f"{ds}_{obj}_time.txt").write_text(
                    "Maximum resident set size (kbytes): 1000000\n")
        st, rss = _a3_ratios(a3)
        check(all(abs(v - 4.0) < 1e-12 for v in st.values())
              and abs(max(rss) - 1.0) < 1e-9,
              "a3 ratios: 200/(10+40) = 4x, RSS 1.0 GB, hand-computed")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "tab_cost.tex"))
    # Interim sourcing exists only for the pre-grid era. Once the grid is
    # on disk, a bare invocation must not silently regress the published
    # grid-sourced macros (this happened once: an a3-macro rebuild without
    # the flag overwrote 47--64x with the interim 22--63x and the wrong
    # range reached two compiled PDFs before the IR1-reconstruction check
    # caught it).
    ap.add_argument("--fit-from-grid", default=None,
                    help="grid root; fit column from grid wall clocks "
                         "(refuses if any of the 20 required cells is absent)")
    ap.add_argument("--t42-window-dir", default=str(T42),
                    help="dir with T4.2 worker window_*.json files "
                         "(overridden only in tests)")
    ap.add_argument("--aux-registry", default=None,
                    help="auxiliary-load window registry "
                         "(default results/aux_windows.json)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.fit_from_grid is None and (CODE_ROOT / "results"
                                    / "P2_main_grid").exists():
        print("REFUSING TO RUN: the grid exists on disk; a bare "
              "invocation would regress the published grid-sourced "
              "macros to interim sourcing. Pass "
              "--fit-from-grid results/P2_main_grid.", file=sys.stderr)
        return 2
    out = build(Path(a.out), Path(a.fit_from_grid) if a.fit_from_grid else None,
                Path(a.t42_window_dir),
                Path(a.aux_registry) if a.aux_registry else None)
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
