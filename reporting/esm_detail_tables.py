"""ESM per-axis detail fragments (P5 SS3.5), four in one generator, each
from its stored artifact, each with schema guards.

    PYTHONHASHSEED=2025 python reporting/esm_detail_tables.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import (display_name, TableStyle,  # noqa: E402
                             code_cell, dataframe_to_tex, write_tex)
from reporting.termmap import data_display                    # noqa: E402

T4F = CODE_ROOT / "results" / "T4_perm_on_sni" / "t4f_sixway_cells.csv"
FIVEPAIRS = CODE_ROOT / "results" / "T3_five_way" / "fiveway_pairs.csv"
FCELLS = CODE_ROOT / "results" / "T3_faithfulness" / "faithfulness_cells.csv"
R1SENS = CODE_ROOT / "results" / "T3_faithfulness" / "r1_threshold_sensitivity.csv"
T2F = CODE_ROOT / "results" / "T2f_d_stability"
OUT = CODE_ROOT / "reporting" / "out"


#: The row-set codes are artifact keys ("rows12", "own16",
#: "by_construction"); printed raw they are undefined strings a reader cannot
#: resolve, and two rows differing only by such a code look like duplicates.
_ROWS_LABEL = {"rows12": "12 common rows", "own16": "its own 16 rows",
               "by_construction": "by construction"}


def _rows_label(v) -> str:
    k = str(v)
    return _ROWS_LABEL.get(k, k.replace("_", " "))


def _prov(gen_note: str, src: Path) -> dict:
    return {"generator": f"reporting/esm_detail_tables.py ({gen_note})",
            "input": str(src), "code_SNI commit": runconfig.git_commit()}


def build_recovery(out_path: Path, src: Path = T4F) -> Path:
    df = pd.read_csv(src)
    need = {"regime", "seed", "method", "auroc", "auprc", "prec_at_k", "shd"}
    assert need.issubset(df.columns), sorted(need - set(df.columns))
    g = (df.groupby(["regime", "method"])
         [["auroc", "auprc", "prec_at_k", "shd"]]
         .agg(["mean", "std"]))
    rows = []
    for (regime, method), r in g.iterrows():
        rows.append({
            "Regime": display_name(regime),
            # Frozen artifact keys ("P-alone") are rendered through the
            # registry, never printed raw (P5R-K SS4.2).
            "Object": data_display(method).replace("_", r"\_")
                      if data_display(method) != method
                      else method.replace("_", r"\_"),
            **{m.upper(): rf"${r[(m, 'mean')]:.3f}"
                          rf"{{\scriptstyle\pm{r[(m, 'std')]:.3f}}}$"
               for m in ("auroc", "auprc", "prec_at_k", "shd")}})
    # T6.1: the same-host probe recomputed on a retrained host, with and
    # without the privileged error signal. Appended rather than substituted --
    # the archived reading stays visible beside the corrected one, and the two
    # effects (retraining, removing the signal) stay separable.
    sym = (CODE_ROOT / "results" / "T6_symmetry"
           / "no_oracle_recovery_cells_own.csv")
    if not sym.exists():
        raise FileNotFoundError(
            f"{sym} is missing: the recovery detail table would then show "
            f"only the reading taken under the privileged error signal.")
    sv = pd.read_csv(sym)
    # Daggers rather than spelled-out variants: the spelled-out labels widened
    # the Object column past the text block.
    LABEL = {"refit_oracle": r"Permutation-on-SNI$^{\dagger}$",
             "refit_no_oracle": r"Permutation-on-SNI$^{\ddagger}$"}
    gs = (sv[(sv.method == "Permutation-on-SNI")
             & (sv.variant.isin(LABEL))]
          .groupby(["regime", "variant"])
          [["auroc", "auprc", "prec_at_k", "shd"]].agg(["mean", "std"]))
    for (regime, variant), r in gs.iterrows():
        rows.append({
            "Regime": display_name(regime),
            "Object": LABEL[variant],
            **{m.upper(): rf"${r[(m, 'mean')]:.3f}"
                          rf"{{\scriptstyle\pm{r[(m, 'std')]:.3f}}}$"
               for m in ("auroc", "auprc", "prec_at_k", "shd")}})
    rows.sort(key=lambda d: (d["Regime"], d["Object"]))

    tex = dataframe_to_tex(
        pd.DataFrame(rows),
        caption=(r"Recovery axis, per-regime detail: mean$\pm$sd over the "
                 r"five synthetic seeds for every object and metric "
                 r"(common-row caliber). Within each regime the two rows marked "
                 r"$\dagger$ and $\ddagger$ are the same-host probe "
                 r"recomputed under information symmetry; the archived row is "
                 r"retained beside them."),
        label="tab:esm_recovery_cells", column_format="llcccc",
        header=["Regime", "Object", "AUROC", "AUPRC", "P@K", "SHD"],
        # the regime column carries readable names now, not keys; 4 pt of
        # column padding buys the width that costs, so the remedy is not an
        # abbreviation back toward a key
        style=TableStyle(environment="table", col_sep_pt=4.0, notes=(
            r"Per-seed values are in the repository artifacts named in the "
            r"provenance header; rows are the six-way comparison's cells. "
            r"\emph{Information symmetry:} The archived "
            r"Permutation-on-SNI row was produced by an ablation whose error "
            r"signal is measured against the values withheld from the "
            r"imputer, which is not true of the objects it is compared with. "
            r"The two added rows separate the two things that changed: "
            r"$\dagger$ a host retrained on the same protocol with that "
            r"signal kept, and $\ddagger$ the same host with the error "
            r"signal taken from its own completed table instead. "
            r"Each variant is scored by the "
            r"archived recipe on its own common rows; the comparison on one "
            r"row set for all variants is in the main text's recovery table "
            r"note.",)),
        escape_data=False)
    # ---- the audit history the main table's note used to carry ---------- #
    # Seventh review SS11.3: one table note was carrying the current reading,
    # the archived reading, the fresh oracle control, the external-host
    # comparison, the XOR saturation check and a superseded verdict at once,
    # so different estimands and different signal calibers were being read
    # together. The decomposition lives here now; the main note keeps the
    # current reading and points at this.
    from reporting.table_recovery import (_host_gap_note, _symmetry_note,
                                          _xor_saturation_note)
    hist = ("\n\n\\paragraph{Recovery axis: audit history.} "
            "This subsection carries the decomposition behind the recovery "
            "table in the main text: what the archived reading was, what "
            "changed when the host was retrained, what changed again when the "
            "error signal was made symmetric, how the symmetric probe compares "
            "with the strongest externally hosted readout (a different "
            "comparison from the same-host one, and not a substitute for it), "
            "and why the XOR regime discriminates poorly among behavioral "
            "readouts. None of it is a second analysis; it is the same cells "
            "read the ways they were read on the way here. "
            # moved verbatim out of the main table's note, ninth review P2-2
            "\\textit{The earlier cell-level Wilcoxon verdict is retained "
            "only as an audit-history artifact and is not used for "
            "inference.} Its arithmetic is no longer computed on this "
            "study's data at all: the generator checks only that the frozen "
            "archived record still reproduces itself (ninth review P0-1).\n\n"
            + _host_gap_note() + " " + _symmetry_note() + " "
            + _xor_saturation_note() + "\n")
    prov = _prov("recovery", src)
    prov["input (symmetry)"] = str(sym)
    return write_tex(out_path, tex + hist, provenance=prov)


def build_repro(out_path: Path, pairs_src: Path = FIVEPAIRS,
                t2f: Path = T2F) -> Path:
    pairs = pd.read_csv(pairs_src)
    assert {"group", "rows", "a", "b", "spearman"}.issubset(pairs.columns)
    agg = (pairs.groupby(["group", "rows"])["spearman"]
           .agg(["mean", "min", "count"]).reset_index())
    def _group(g: str) -> str:
        # "MIMIC|P" -> "MIMIC / TAP": the second field is a frozen method key.
        parts = [data_display(x) if data_display(x) != x
                 else x.replace("_", r"\_") for x in str(g).split("|")]
        return " / ".join(parts)

    rows = [{"Object (dataset)": _group(r.group),
             "Rows": _rows_label(r.rows),
             "Pairs": int(r["count"]),
             r"$\rho$ mean": f"{r['mean']:.4f}",
             r"$\rho$ min": f"{r['min']:.4f}"}
            for _, r in agg.iterrows()]
    # T6.1: the sixth object -- the same-host behavioral readout, which IS
    # the host band the main text compares D against -- with and without the
    # privileged error signal. Both are shown: the corrected band is the one
    # that belongs in the comparison, and the archived one is what the
    # submitted version reported.
    import json
    bandf = (CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_band.json")
    if not bandf.exists():
        raise FileNotFoundError(
            f"{bandf} is missing: this table would then omit the host band "
            f"entirely, or show only the reading taken under the privileged "
            f"error signal.")
    band = json.loads(bandf.read_text())
    for ds, d in band["datasets"].items():
        for key, lab in (("band_oracle", "Permutation-on-SNI (same host, "
                                         "archived signal)"),
                         ("band_noOracle", "Permutation-on-SNI (same host, "
                                           "no oracle)")):
            b = d[key]
            rows.append({"Object (dataset)": f"{ds} / {lab}",
                         "Rows": "seed-1 observed rows",
                         "Pairs": int(b["n_pairs"]),
                         r"$\rho$ mean": f"{b['mean']:.4f}",
                         r"$\rho$ min": f"{b['min']:.4f}"})

    t1 = dataframe_to_tex(
        pd.DataFrame(rows),
        caption=(r"Reproducibility axis: cross-seed pairwise Spearman per "
                 r"object and row caliber (all pairs). The last four rows are "
                 r"the same-host behavioral readout that supplies the host "
                 r"band, before and after the information-symmetry "
                 r"correction."),
        label="tab:esm_repro_pairs", column_format="llccc",
        header=["Object (dataset)", "Rows", "Pairs", r"$\rho$ mean",
                r"$\rho$ min"],
        style=TableStyle(environment="table", col_sep_pt=4.0, notes=(
            r"\emph{Information symmetry:} The same-host readout's "
            r"ablation measured its error against the values withheld from "
            r"the imputer. With that privileged signal the error is anchored "
            r"to fixed withheld values while the host's predictions move with "
            r"the seed, so the readout absorbs seed noise; without it both "
            r"sides move with the host together and the readout is "
            r"correspondingly more stable. The corrected band is the one the "
            r"main text compares \Dm{} against; the archived band is "
            r"retained here rather than replaced.",)),
        escape_data=False)

    # Thread-perturbation arms (B84 family): rho of D under 1/8/24 BLAS
    # threads against the pinned t2 reference, computed from the stored
    # matrices.
    from scipy.stats import spearmanr
    ref = pd.read_csv(t2f / "D_MIMIC_seed1_cpu_t2.csv", index_col=0)
    off = ~np.eye(len(ref), dtype=bool)
    prow = []
    for k in (1, 8, 24):
        M = pd.read_csv(t2f / f"D_MIMIC_perturb_cpu_t{k}.csv", index_col=0)
        assert list(M.index) == list(ref.index), f"t{k}: row mismatch"
        rho = float(spearmanr(ref.to_numpy(float)[off],
                              M.to_numpy(float)[off]).statistic)
        prow.append({"Arm": f"BLAS threads = {k}",
                     r"$\rho$ vs pinned t2": f"{rho:.4f}"})
    t2 = dataframe_to_tex(
        pd.DataFrame(prow),
        caption=(r"Thread-perturbation arms (MIMIC, seed 1): rank agreement "
                 r"of \Dm{} against the pinned two-thread reference."),
        label="tab:esm_repro_perturb", column_format="lc",
        header=["Arm", r"$\rho$ vs pinned t2"],
        style=TableStyle(environment="table", notes=(
            r"The thread count is a controlled variable (finding B84): an "
            r"unpinned BLAS changes the trained model, hence \Dm{}.",)),
        escape_data=False)
    from reporting.table_fiveway_stability import band_control_note
    moved = ("\n\n\\paragraph{Stability axis: what the main table's note "
             "points at.} The two same-host rows of the main text's "
             "cross-seed stability table differ only in the error signal they "
             "were scored against. Why the band moves between them is the "
             "note to Table~\\ref{tab:esm_repro_pairs} above, which already "
             "states it; the main table's note carried a second copy of that "
             "explanation and no longer does (ninth review P2-2). What "
             "follows is the control that was in the main note and is not "
             "stated anywhere else.\n\n"
             + band_control_note() + "\n")
    return write_tex(out_path, t1 + "\n\n" + t2 + moved,
                     provenance=_prov("repro", pairs_src))


def build_faith_targets(out_path: Path, src: Path = FCELLS) -> Path:
    df = pd.read_csv(src)
    df = df[(df.scope == "full") & df.method.isin(["SNI-D", "P"])]
    assert len(df), "no full-scope SNI-D/P rows"
    rows = []
    for (ds, tgt), g in df.groupby(["dataset", "target"]):
        d_med = g[g.method == "SNI-D"].rho.median()
        p_med = g[g.method == "P"].rho.median()
        dd = (g[g.method == "SNI-D"].set_index("seed").rho
              - g[g.method == "P"].set_index("seed").rho)
        rows.append({"Dataset": ds, "Target": code_cell(tgt),
                     r"\Dm{} median $\rho$": f"{d_med:.3f}",
                     r"\TAP{} median $\rho$": f"{p_med:.3f}",
                     r"$\Delta$ median": f"{dd.median():.3f}",
                     r"$\Delta$ range": rf"[{dd.min():.3f}, {dd.max():.3f}]"})
    tex = dataframe_to_tex(
        pd.DataFrame(rows).sort_values(["Dataset", "Target"]),
        caption=(r"Faithfulness axis, per-target detail (five-seed subset, all "
                 r"twelve targets): median row-level $\rho$ against the "
                 r"ablation matrix over the original five seeds, per target; "
                 r"$\Delta = $ \Dm{} $-$ \TAP{}."),
        label="tab:esm_faith_targets", column_format="llcccc",
        header=["Dataset", "Target", r"\Dm{} med.\ $\rho$",
                r"\TAP{} med.\ $\rho$", r"$\Delta$ med.", r"$\Delta$ range"],
        style=TableStyle(environment="table", notes=(
            r"Per-target $n=5$ seeds is too thin for a per-target interval; "
            r"family-level effect sizes and bootstrap CIs are in the main "
            r"text's faithfulness statistics and the reliability-ceiling analysis artifact.",)),
        escape_data=False)
    return write_tex(out_path, tex, provenance=_prov("faith-targets", src))


def build_r1(out_path: Path,
             src: Path = CODE_ROOT / "results" / "T5_stats"
             / "t51_redundancy_sensitivity.json",
             names_src: Path = R1SENS) -> Path:
    """P5R-H SS2: the sensitivity table under the unified T estimand
    (15 seeds, exact enumeration, seed-only bootstrap CI); the excluded
    cluster names come from the original pre-check artifact."""
    from experiments.t51_cluster_stats import seed_boot_ci_T
    cells = json.loads(src.read_text())["cells"]
    names = {}
    ndf = pd.read_csv(names_src)
    for _, r in ndf.iterrows():
        # A threshold that excludes nothing has an empty R; pandas reads the
        # blank cell as NaN and str() would print the word "nan" into the
        # table (P5R-K SS4.2, found by page inspection).
        raw = r.get("R", "")
        txt = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) \
            else str(raw).strip()
        names[(r["dataset"], f"{float(r['tau']):g}")] = txt
    rows = []
    for key, c in sorted(cells.items()):
        ds, tau = key.split("|")
        tau = tau.split("=")[1]
        if c.get("m1"):
            m1 = c["m1"]
            med = {int(k): [v] for k, v in
                   c["aggregates"]["seed_medians"].items()}
            lo, hi = seed_boot_ci_T(med)
            neg = sum(1 for v in c["aggregates"]["seed_medians"].values()
                      if v < 0)
            t_txt = f"{m1['observed_stat_mean_of_block_medians']:+.3f}"
            p_txt = f"{m1['p_two_sided']:.4f}"
            ci_txt = f"[{lo:+.3f}, {hi:+.3f}]"
            neg_txt = f"{neg}/{c['n_seeds']}"
        else:
            t_txt = p_txt = ci_txt = neg_txt = "--"
        rows.append({
            "Data": ds, r"$\tau$": tau, r"$|R|$": c["R_size"],
            "keep": c["n_keep"],
            "appl.": r"\checkmark" if c["applicable"] else "--",
            r"$T$": t_txt,
            r"exact $p$": p_txt,
            r"CI$_{95}$": ci_txt,
            r"seeds$<0$": neg_txt,
            "Excluded set": "; ".join(
                code_cell(x) for x in
                (names.get((ds, tau), "") or "none").split(";"))})
    body = pd.DataFrame(rows)
    tex = dataframe_to_tex(
        body, caption=(r"Redundancy pre-check: sensitivity of the "
                       r"faithfulness comparison to the exclusion "
                       r"threshold $\tau$ (unified $T$ estimand, 15 "
                       r"seeds)."),
        label="tab:esm_r1_sensitivity",
        column_format="l" + "c" * (len(body.columns) - 2)
                      + r"p{0.16\linewidth}",
        header=list(body.columns),
        style=TableStyle(environment="table*", notes=(
            r"A possible explanation --- stated as a hypothesis, not a "
            r"finding: permutation ablations are compressed within "
            r"redundant clusters (permuting one column of the "
            r"blood-pressure family lets the model recover the signal from "
            r"the rest), so ablation deltas approach noise there and all "
            r"products' rankings converge toward random; removing the "
            r"cluster exposes the informative region where \TAP{}'s "
            r"advantage shows more clearly. $T$ = mean of seed-level "
            r"medians; exact enumeration under seed-block "
            r"sign-exchangeability; CI = seed-only bootstrap -- the same "
            r"estimand as every faithfulness statistic. The historical "
            r"pair-level values remain in the archived artifacts.",)),
        escape_data=False)
    return write_tex(out_path, tex, provenance=_prov("r1-sensitivity",
                                                     src))


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        p1 = build_recovery(t / "a.tex")
        check(r"\pm" in p1.read_text() and "linear\\_gaussian"
              in p1.read_text(), "recovery: mean+-sd cells, escaped regimes")
        p2 = build_repro(t / "b.tex")
        tx = p2.read_text()
        check("BLAS threads = 24" in tx and "B84" in tx,
              "repro: perturbation arms with B84 note")
        p3 = build_faith_targets(t / "c.tex")
        check(r"$\Delta$ range" in p3.read_text(),
              "faith targets: delta range column")
        p4 = build_r1(t / "d.tex")
        check("hypothesis, not a" in p4.read_text(),
              "r1: hypothesis sentence marked as hypothesis")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    for fn, name in ((build_recovery, "sec_esm_recovery.tex"),
                     (build_repro, "sec_esm_repro.tex"),
                     (build_faith_targets, "sec_esm_faith_targets.tex"),
                     (build_r1, "sec_esm_r1.tex")):
        out = fn(OUT / name)
        print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
