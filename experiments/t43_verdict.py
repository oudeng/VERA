"""T4.3 mechanical readout: parent tiers + interpretation supplement.

Implements, verbatim and by import rather than reimplementation:
  * the parent verdict rule  docs/T43_noprior_decision_rule.md (6535787);
  * the interpretation supplement docs/T43_interpretation_supplement.md
    (0be7e25): observables O1-O4 with recompute-asserted calibers, the
    mechanism ladder (MECH-PRIOR / MECH-OPTIMIZED / MECH-MIXED), the O4
    reading bands (non-gating);
  * P4K-A filings: (1) if MECH-MIXED is triggered by O2 alone, the
    with-prior D's own absolute level on the same table is attached
    (runtime-read); (2) NoPrior's own host-reproducibility band is always
    attached (stage_real's algorithm on the NP ablation matrices).

Every reference value is read from stored artifacts at runtime and every
caliber is proven by recomputing a with-prior quantity and asserting it
against its stored value before the NoPrior analogue is computed.

    env PYTHONHASHSEED=2025 python experiments/t43_verdict.py --stage selftest
    env PYTHONHASHSEED=2025 python experiments/t43_verdict.py --stage verdict
"""
from __future__ import annotations

import os

_NT = os.environ.get("SNI_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = CODE_ROOT / "results" / "T4_noprior"
FAITH = CODE_ROOT / "results" / "T3_faithfulness"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
FIVEWAY = CODE_ROOT / "results" / "T3_five_way"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
STAB = CODE_ROOT / "results" / "T2f_d_stability"

DS = "eICU"                       # the axis's single table (supplement scope)
SEEDS = [1, 2, 3, 5, 8]
REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SYNTH_SEEDS = [2025, 2026, 2027, 2028, 2029]
ALPHA = 0.05
FLOOR = 0.30                      # parent rule's absolute-faithfulness floor


# --------------------------------------------------------------------------- #
# pure decision functions (exhaustively unit-tested in --stage selftest)
# --------------------------------------------------------------------------- #
def wilcoxon_p(vec: np.ndarray) -> float:
    """Same semantics as faithfulness.stage_analyze's guard."""
    from scipy.stats import wilcoxon
    v = np.asarray(vec, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    if np.allclose(v, 0.0):
        return 1.0
    return float(wilcoxon(v).pvalue)


def parent_tier(faith_med_delta: float, faith_p: float,
                rec_med_delta: float, rec_p: float) -> str:
    """6535787's three tiers, precedence top-down."""
    if (faith_med_delta > 0 and faith_p < ALPHA) or \
       (rec_med_delta > 0 and rec_p < ALPHA):
        return "DIFFERENT-TYPE"
    if faith_med_delta > 0 or rec_med_delta > 0:
        return "BORDERLINE"
    return "SAME-TYPE"


def mechanism(tier: str, o1: float, o2: float, t_stab: float) -> tuple:
    """Supplement's ladder. Returns (verdict, deviations)."""
    if tier == "DIFFERENT-TYPE":
        return "MECH-PRIOR", []
    dev = []
    if tier != "SAME-TYPE":
        dev.append(f"O3={tier}")
    if not o1 >= t_stab:
        dev.append(f"O1={o1:.4f} < T_stab={t_stab:.4f}")
    if not o2 < FLOOR:
        dev.append(f"O2={o2:.4f} >= floor={FLOOR}")
    if not dev:
        return "MECH-OPTIMIZED", []
    return "MECH-MIXED", dev


def o4_band(o4: float, rho_prior: float) -> str:
    # 1e-12 guards the real-number semantics of ">=" at the band edge
    # against binary-FP representation (0.80 - 0.10 > 0.70 in floats).
    if o4 >= rho_prior - 0.10 - 1e-12:
        return "A"
    if o4 <= 0.50 + 1e-12:
        return "B"
    return "intermediate"


# --------------------------------------------------------------------------- #
# caliber helpers (imported from their owners, thin wrappers only)
# --------------------------------------------------------------------------- #
def _stability_mean(mats: dict) -> float:
    """five_way caliber: mean pairwise Spearman, own rows (all), off-diag."""
    from five_way_stability import _pairs_rows
    keep = np.ones(len(next(iter(mats.values())).index), dtype=bool)
    rows = _pairs_rows(mats, keep, "t43", "own")
    return float(np.mean([r["spearman"] for r in rows]))


def _agreement_mean(mats: dict, P: pd.DataFrame) -> float:
    """five_way caliber: per-seed Spearman(_vec(M), _vec(P)), mean."""
    from scipy.stats import spearmanr
    from five_way_stability import _vec
    keep = np.ones(len(P.index), dtype=bool)
    vals = [float(spearmanr(_vec(M.fillna(0.0), keep), _vec(P, keep)).statistic)
            for M in mats.values()]
    return float(np.mean(vals))


def _host_band_pairs(A_by_seed: dict) -> list:
    """t4f_perm_on_sni.stage_real's algorithm, verbatim caliber: pairwise
    Spearman on the entries selected by the FIRST seed's notna mask."""
    from scipy.stats import spearmanr
    seeds = sorted(A_by_seed)
    first = A_by_seed[seeds[0]]
    sel = first.notna().to_numpy()
    rows = []
    for a, b in combinations(seeds, 2):
        A = A_by_seed[a].to_numpy(float)[sel]
        B = A_by_seed[b].to_numpy(float)[sel]
        r = spearmanr(A, B)
        rows.append({"a": a, "b": b, "spearman": round(float(r.statistic), 4)})
    return rows


def _row_rhos(M: pd.DataFrame, A: pd.DataFrame) -> dict:
    """faithfulness caliber: Spearman(M row, A row) over sources != target."""
    from scipy.stats import spearmanr
    feats = list(M.index)
    out = {}
    for f in A.index:
        srcs = [c for c in feats if c != f]
        out[f] = float(spearmanr(M.loc[f, srcs].to_numpy(float),
                                 A.loc[f, srcs].to_numpy(float)).statistic)
    return out


def _assert_close(name: str, got: float, ref: float, tol: float,
                  log: list) -> None:
    ok = abs(got - ref) <= tol
    log.append({"check": name, "recomputed": round(got, 6),
                "stored": ref, "tol": tol, "pass": bool(ok)})
    if not ok:
        raise AssertionError(
            f"caliber check failed: {name}: recomputed {got:.6f} vs stored "
            f"{ref} (tol {tol}) -- readout stops, adjudicate before reading "
            f"any NoPrior number.")


# --------------------------------------------------------------------------- #
def stage_verdict() -> int:
    from faithfulness import _paired_effect
    from pilot_r21 import measured_rows, score
    from t43_noprior_synth import run_cell as _synth_cell  # noqa: F401 (path check)

    checks: list = []
    src: dict = {}

    # ---- references, runtime-read ---------------------------------------- #
    fw = json.loads((FIVEWAY / "fiveway_summary.json").read_text())
    ref_stab = float(fw[DS]["SNI-D"]["stability_mean"])
    ref_agree = float(fw[DS]["SNI-D"]["rho_with_P_mean"])
    src["ref_stability_withprior"] = str(FIVEWAY / "fiveway_summary.json")
    hostcsv = pd.read_csv(T4F / "perm_on_sni_real_stability.csv")
    host_rows = hostcsv[hostcsv.dataset == DS]
    host = float(host_rows.spearman.mean())
    src["host_band"] = str(T4F / "perm_on_sni_real_stability.csv")

    # ---- caliber proofs (with-prior recomputations) ----------------------- #
    D_arch = {s: pd.read_csv(STAB / f"D_{DS}_seed{s}_cpu_t2.csv", index_col=0)
              for s in SEEDS}
    _assert_close("stability caliber vs fiveway (archived T2f D)",
                  _stability_mean(D_arch), ref_stab, 1e-3, checks)
    D_retr = {s: pd.read_csv(FAITH / f"D_retrained_{DS}_seed{s}_cpu_t2.csv",
                             index_col=0) for s in SEEDS}
    _assert_close("stability caliber vs fiveway (retrained D, as committed; "
                  "known eps-level CSV-roundtrip diff 9.7e-17)",
                  _stability_mean(D_retr), ref_stab, 1e-3, checks)
    feats = list(D_arch[SEEDS[0]].index)
    P = pd.read_csv(PRIOR / f"P_{DS}_seed1_cpu_t2.csv", index_col=0
                    ).reindex(index=feats, columns=feats)
    src["P"] = str(PRIOR / f"P_{DS}_seed1_cpu_t2.csv")
    _assert_close("agreement caliber vs fiveway (archived D)",
                  _agreement_mean(D_arch, P), ref_agree, 1e-3, checks)
    A_prior = {s: pd.read_csv(FAITH / f"A_{DS}_seed{s}_cpu_t2.csv", index_col=0)
               for s in SEEDS}
    got_pairs = {(r["a"], r["b"]): r["spearman"]
                 for r in _host_band_pairs(A_prior)}
    for _, r in host_rows.iterrows():
        _assert_close(f"host-band pair ({int(r.a)},{int(r.b)}) vs stage_real",
                      got_pairs[(int(r.a), int(r.b))], float(r.spearman),
                      1e-3, checks)

    # ---- faithfulness axis (decisive) ------------------------------------ #
    A_np = {s: pd.read_csv(FAITH / f"A_NP_{DS}_seed{s}_cpu_t{_NT}.csv",
                           index_col=0) for s in SEEDS}
    D_np = {s: pd.read_csv(FAITH / f"D_retrained_NP_{DS}_seed{s}_cpu_t{_NT}.csv",
                           index_col=0) for s in SEEDS}
    src["A_NP"] = str(FAITH / f"A_NP_{DS}_seed*_cpu_t{_NT}.csv")
    rho_np, rho_tap, deltas = [], [], []
    for s in SEEDS:
        rd = _row_rhos(D_np[s], A_np[s])
        rp = _row_rhos(P, A_np[s])
        for f in rd:
            rho_np.append(rd[f])
            rho_tap.append(rp[f])
            deltas.append(rd[f] - rp[f])
    n_pairs = len(deltas)
    assert n_pairs == 60, f"faithfulness axis expects 60 pairs, got {n_pairs}"
    faith = {"n_pairs": n_pairs,
             "median_rho_NP": float(np.median(rho_np)),
             "median_rho_TAP": float(np.median(rho_tap)),
             "median_delta": float(np.median(deltas)),
             "wilcoxon_p": wilcoxon_p(np.array(deltas)),
             **_paired_effect(np.array(deltas))}

    # ---- recovery axis ---------------------------------------------------- #
    from t43_noprior_synth import PILOT_METHODS, PILOT
    cells = pd.read_csv(OUT / "t43_noprior_synth_cells.csv")
    assert len(cells) == 15, f"recovery axis expects 15 cells, got {len(cells)}"
    from pilot_r21 import load_cell
    rec_np, rec_tap, rec_deltas, cell_rows = [], [], [], []
    for regime in REGIMES:
        for s in SYNTH_SEEDS:
            complete, *_rest = load_cell(regime, s)
            G = _rest[2]
            cols = list(complete.columns)
            D = pd.read_csv(OUT / f"D_NP_{regime}_s{s}.csv", index_col=0
                            ).reindex(index=cols, columns=cols).fillna(0.0)
            mats = {m: pd.read_csv(PILOT / f"D_{regime}_s{s}_{m}.csv",
                                   index_col=0).reindex(index=cols,
                                                        columns=cols).fillna(0.0)
                    for m in PILOT_METHODS}
            common = np.ones(len(cols), dtype=bool)
            for M in mats.values():
                common &= measured_rows(M)
            common &= measured_rows(D)
            sc_np = score(D, G, keep=common)
            stored = cells[(cells.regime == regime) & (cells.seed == s)]
            _assert_close(f"recovery recompute {regime}/s{s} vs stored cells "
                          f"csv", float(sc_np["auroc"]),
                          float(stored.NP_auroc.iloc[0]), 1.01e-4, checks)
            Pm = pd.read_csv(PRIOR / f"P_synth_{regime}_s{s}.csv", index_col=0
                             ).reindex(index=cols, columns=cols).fillna(0.0)
            sc_tap = score(Pm, G, keep=common)
            rec_np.append(float(sc_np["auroc"]))
            rec_tap.append(float(sc_tap["auroc"]))
            rec_deltas.append(rec_np[-1] - rec_tap[-1])
            cell_rows.append({"regime": regime, "seed": s,
                              "NP_auroc": rec_np[-1], "TAP_auroc": rec_tap[-1],
                              "n_rows": int(common.sum())})
    rec = {"n_cells": len(rec_deltas),
           "median_delta": float(np.median(rec_deltas)),
           "wilcoxon_p": wilcoxon_p(np.array(rec_deltas)),
           **_paired_effect(np.array(rec_deltas))}

    # ---- O1-O4, thresholds, verdicts -------------------------------------- #
    o1 = _stability_mean(D_np)
    mf12 = pd.read_csv(FIVEWAY /
                       f"D_{DS}_seed1_MissForest-importance.csv", index_col=0)
    meas12 = (~mf12.reindex(index=feats, columns=feats).isna().all(axis=1)
              ).to_numpy()
    from five_way_stability import _pairs_rows
    o1_rows12 = float(np.mean([r["spearman"] for r in
                               _pairs_rows(D_np, meas12, "t43", "rows12")]))
    o2 = faith["median_rho_NP"]
    o4 = _agreement_mean(D_np, P)
    t_stab = host + 0.5 * (ref_stab - host)
    tier = parent_tier(faith["median_delta"], faith["wilcoxon_p"],
                       rec["median_delta"], rec["wilcoxon_p"])
    mech, dev = mechanism(tier, o1, o2, t_stab)
    band = o4_band(o4, ref_agree)

    # P4K-A filing 1: O2-only MECH-MIXED carries the with-prior fact.
    filing1 = None
    if mech == "MECH-MIXED" and all(d.startswith("O2") for d in dev):
        fs = json.loads((FAITH / "faithfulness_summary.json").read_text())
        filing1 = {"withprior_D_rho_median_same_table":
                   float(fs[DS]["SNI-D"]["rho_median"]),
                   "source": str(FAITH / "faithfulness_summary.json"),
                   "note": "with-prior D itself is above the 0.30 floor on "
                           "this table; MECH-MIXED here reflects the "
                           "pre-recorded conservative bias of the absolute "
                           "floor (P4K-A filing 1), wording stays narrow."}
    # P4K-A filing 2: NoPrior's own host band, always attached.
    np_host_pairs = _host_band_pairs(A_np)
    np_host = float(np.mean([r["spearman"] for r in np_host_pairs]))

    out = {
        "rule_commits": {"parent": "6535787", "supplement": "0be7e25"},
        "scope": {"table": DS, "note": "single-table control; MIMIC untested "
                                       "for the NoPrior variant (binding "
                                       "wording, supplement)"},
        "caliber_checks": checks,
        "sources": src,
        "faithfulness_axis": faith,
        "recovery_axis": rec,
        "recovery_cells": cell_rows,
        "observables": {"O1_stability_own": round(o1, 4),
                        "O1_stability_rows12": round(o1_rows12, 4),
                        "O2_median_rho": round(o2, 4),
                        "O3_parent_tier": tier,
                        "O4_rho_with_TAP": round(o4, 4)},
        "thresholds": {"host_band_withprior": round(host, 4),
                       "withprior_stability": ref_stab,
                       "T_stab": round(t_stab, 4),
                       "floor": FLOOR,
                       "withprior_rho_with_TAP": ref_agree},
        "parent_verdict": tier,
        "mechanism_verdict": mech,
        "mechanism_deviations": dev,
        "O4_reading_band": band,
        "filing1_O2_bias": filing1,
        "filing2_noprior_own_host_band": {
            "mean": round(np_host, 4), "pairs": np_host_pairs,
            "caliber": "t4f stage_real algorithm on A_NP matrices"},
    }
    (OUT / "t43_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("parent_verdict", "mechanism_verdict",
                       "mechanism_deviations", "O4_reading_band",
                       "observables", "thresholds")}, indent=1))
    print(f"[ok] wrote {OUT / 't43_verdict.json'} "
          f"({len(checks)} caliber checks passed)")
    return 0


# --------------------------------------------------------------------------- #
def stage_observables_15seed() -> int:
    """Chat ruling 2026-08-29 extra 3: descriptive 15-seed refresh of the
    O1-O4 mechanism panel (pure readout, zero training). Thresholds and
    category rules are the prospectively specified ones, unchanged; if any category
    (parent tier, mechanism verdict, O4 band) differs from the shipped
    5-seed verdict, this stage STOPS the panel update by reporting the
    flip loudly and writing nothing for prose to cite."""
    from t51_cluster_stats import EXP_SEEDS
    seeds15 = SEEDS + EXP_SEEDS
    ship = json.loads((OUT / "t43_verdict.json").read_text())

    fw = json.loads((FIVEWAY / "fiveway_summary.json").read_text())
    ref_stab = float(fw[DS]["SNI-D"]["stability_mean"])
    ref_agree = float(fw[DS]["SNI-D"]["rho_with_P_mean"])
    hostcsv = pd.read_csv(T4F / "perm_on_sni_real_stability.csv")
    host = float(hostcsv[hostcsv.dataset == DS].spearman.mean())
    t_stab = host + 0.5 * (ref_stab - host)
    assert abs(t_stab - float(ship["thresholds"]["T_stab"])) <= 5.1e-5, \
        "threshold drift vs shipped verdict"

    D_np = {s: pd.read_csv(FAITH / f"D_retrained_NP_{DS}_seed{s}_cpu_t{_NT}"
                           f".csv", index_col=0) for s in seeds15}
    A_np = {s: pd.read_csv(FAITH / f"A_NP_{DS}_seed{s}_cpu_t{_NT}.csv",
                           index_col=0) for s in seeds15}
    feats = list(D_np[SEEDS[0]].index)
    P = pd.read_csv(PRIOR / f"P_{DS}_seed1_cpu_t2.csv", index_col=0
                    ).reindex(index=feats, columns=feats)

    o1 = _stability_mean(D_np)
    mf12 = pd.read_csv(FIVEWAY / f"D_{DS}_seed1_MissForest-importance.csv",
                       index_col=0)
    meas12 = (~mf12.reindex(index=feats, columns=feats).isna().all(axis=1)
              ).to_numpy()
    from five_way_stability import _pairs_rows
    o1_rows12 = float(np.mean([r["spearman"] for r in
                               _pairs_rows(D_np, meas12, "t43", "rows12")]))
    rho_np = [v for s in seeds15
              for v in _row_rhos(D_np[s], A_np[s]).values()]
    o2 = float(np.median(rho_np))
    o4 = _agreement_mean(D_np, P)
    np_host = float(np.mean([r["spearman"]
                             for r in _host_band_pairs(A_np)]))

    # categories under the unchanged prospectively specified rules; the axis medians
    # come from the cluster-robust 15-seed readout (both negative).
    t51x = json.loads((CODE_ROOT / "results" / "T5_stats"
                       / "t51_expanded_15seed.json").read_text())
    faith_eff = float(t51x["families"]["noprior_faithfulness|eICU"]["m1"]
                      ["effect_pooled_median"])
    faith_p = float(t51x["families"]["noprior_faithfulness|eICU"]["m1"]
                    ["p_two_sided"])
    rec = ship["recovery_axis"]
    tier = parent_tier(faith_eff, faith_p,
                       float(rec["median_delta"]), float(rec["wilcoxon_p"]))
    mech, dev = mechanism(tier, o1, o2, t_stab)
    band = o4_band(o4, ref_agree)

    old = {"tier": ship["parent_verdict"],
           "mech": ship["mechanism_verdict"],
           "band": ship["O4_reading_band"]}
    new = {"tier": tier, "mech": mech, "band": band}
    flip = {k: (old[k], new[k]) for k in old if old[k] != new[k]}
    if flip:
        print(f"CATEGORY FLIP -- STOP, report to Chat, no panel update: "
              f"{flip}", file=sys.stderr)
        return 3
    out = {"note": "descriptive 15-seed refresh (Chat ruling 2026-08-29 "
                   "extra 3); categories unchanged vs the shipped 5-seed "
                   "verdict; 5-seed originals remain in the ESM",
           "n_seeds": len(seeds15),
           "observables_15seed": {
               "O1_stability_own": round(o1, 4),
               "O1_stability_rows12": round(o1_rows12, 4),
               "O2_median_rho": round(o2, 4),
               "O3_parent_tier": tier,
               "O4_rho_with_TAP": round(o4, 4)},
           "noprior_own_host_band_mean_15seed": round(np_host, 4),
           "observables_5seed_shipped": ship["observables"],
           "categories": new,
           "mechanism_deviations_15seed": dev,
           "thresholds": ship["thresholds"]}
    (OUT / "t43_observables_15seed.json").write_text(json.dumps(out,
                                                                indent=1))
    print(f"[ok] categories unchanged ({new}); O1 {ship['observables']['O1_stability_own']}"
          f"->{out['observables_15seed']['O1_stability_own']}, "
          f"O2 {ship['observables']['O2_median_rho']}"
          f"->{out['observables_15seed']['O2_median_rho']}, "
          f"O4 {ship['observables']['O4_rho_with_TAP']}"
          f"->{out['observables_15seed']['O4_rho_with_TAP']}; "
          f"deviations: {dev}")
    print(f"[ok] wrote {OUT / 't43_observables_15seed.json'}")
    return 0


def stage_mimic_np_descriptive() -> int:
    """Chat ruling 2026-08-29 (noon) item 2: O1/O4 recomputed
    DESCRIPTIVELY on the 15 NP-MIMIC D matrices (pure readout, zero
    training). No band, no category, no verdict: the mechanism-verdict
    machinery stays anchored to the committed eICU rules. Output goes to
    the ESM marked descriptive."""
    from t51_cluster_stats import EXP_SEEDS
    seeds15 = SEEDS + EXP_SEEDS
    ds = "MIMIC"
    D_np = {s: pd.read_csv(FAITH / f"D_retrained_NP_{ds}_seed{s}_cpu_t{_NT}"
                           f".csv", index_col=0) for s in seeds15}
    A_np = {s: pd.read_csv(FAITH / f"A_NP_{ds}_seed{s}_cpu_t{_NT}.csv",
                           index_col=0) for s in seeds15}
    feats = list(D_np[SEEDS[0]].index)
    P = pd.read_csv(PRIOR / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                    ).reindex(index=feats, columns=feats)
    o1 = _stability_mean(D_np)
    mf12 = pd.read_csv(FIVEWAY / f"D_{ds}_seed1_MissForest-importance.csv",
                       index_col=0)
    meas12 = (~mf12.reindex(index=feats, columns=feats).isna().all(axis=1)
              ).to_numpy()
    from five_way_stability import _pairs_rows
    o1_rows12 = float(np.mean([r["spearman"] for r in
                               _pairs_rows(D_np, meas12, "t43", "rows12")]))
    o4 = _agreement_mean(D_np, P)
    np_host = float(np.mean([r["spearman"]
                             for r in _host_band_pairs(A_np)]))
    fw = json.loads((FIVEWAY / "fiveway_summary.json").read_text())
    out = {"note": "DESCRIPTIVE ONLY (Chat ruling 2026-08-29 item 2): "
                   "no band, no category, no verdict; the mechanism "
                   "machinery stays anchored to the committed eICU rules",
           "dataset": ds, "n_seeds": len(seeds15),
           "O1_stability_own": round(o1, 4),
           "O1_stability_rows12": round(o1_rows12, 4),
           "O4_rho_with_TAP": round(o4, 4),
           "noprior_own_host_band_mean": round(np_host, 4),
           "withprior_reference": {
               "stability_mean": round(float(
                   fw[ds]["SNI-D"]["stability_mean"]), 4),
               "rho_with_P_mean": round(float(
                   fw[ds]["SNI-D"]["rho_with_P_mean"]), 4)}}
    (OUT / "t43_mimic_np_descriptive.json").write_text(
        json.dumps(out, indent=1))
    print(f"[ok] MIMIC NoPrior descriptive: O1={out['O1_stability_own']} "
          f"(with-prior {out['withprior_reference']['stability_mean']}), "
          f"O4={out['O4_rho_with_TAP']} "
          f"(with-prior {out['withprior_reference']['rho_with_P_mean']}), "
          f"own host band {out['noprior_own_host_band_mean']}")
    print(f"[ok] wrote {OUT / 't43_mimic_np_descriptive.json'}")
    return 0


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    # parent tiers, every branch
    check(parent_tier(0.1, 0.01, -0.1, 0.9) == "DIFFERENT-TYPE",
          "tier: faith axis significant win -> DIFFERENT-TYPE")
    check(parent_tier(-0.1, 0.9, 0.1, 0.01) == "DIFFERENT-TYPE",
          "tier: recovery axis significant win -> DIFFERENT-TYPE")
    check(parent_tier(0.1, 0.20, -0.1, 0.9) == "BORDERLINE",
          "tier: positive without significance -> BORDERLINE")
    check(parent_tier(-0.01, 0.9, 0.0, 0.9) == "SAME-TYPE",
          "tier: both medians <= 0 -> SAME-TYPE")
    check(parent_tier(0.1, 0.05, -0.1, 0.9) == "BORDERLINE",
          "tier: p exactly at alpha is NOT significant (strict <)")

    # mechanism ladder, every branch
    check(mechanism("DIFFERENT-TYPE", 0.9, 0.1, 0.7)[0] == "MECH-PRIOR",
          "mech: DIFFERENT-TYPE -> MECH-PRIOR regardless of O1/O2")
    check(mechanism("SAME-TYPE", 0.71, 0.29, 0.7)[0] == "MECH-OPTIMIZED",
          "mech: SAME + over-stable + unfaithful -> MECH-OPTIMIZED")
    m, d = mechanism("SAME-TYPE", 0.69, 0.29, 0.7)
    check(m == "MECH-MIXED" and d == ["O1=0.6900 < T_stab=0.7000"],
          "mech: O1 below T_stab -> MIXED with the deviation named")
    m, d = mechanism("SAME-TYPE", 0.71, 0.31, 0.7)
    check(m == "MECH-MIXED" and d == ["O2=0.3100 >= floor=0.3"],
          "mech: O2 at/above floor -> MIXED, O2-only deviation")
    check(mechanism("BORDERLINE", 0.9, 0.1, 0.7)[0] == "MECH-MIXED",
          "mech: BORDERLINE lands MIXED by construction")
    check(mechanism("SAME-TYPE", 0.70, 0.29, 0.7)[0] == "MECH-OPTIMIZED",
          "mech: O1 exactly at T_stab counts as over-stable (>=)")
    check(mechanism("SAME-TYPE", 0.71, 0.30, 0.7)[1] == ["O2=0.3000 >= floor=0.3"],
          "mech: O2 exactly at floor fails the strict < floor")

    # O4 bands
    check(o4_band(0.75, 0.80) == "A", "band: within 0.10 of prior -> A")
    check(o4_band(0.699, 0.80) == "intermediate",
          "band: just under prior-0.10 -> intermediate")
    check(o4_band(0.50, 0.80) == "B", "band: 0.50 exactly -> B (<=)")
    check(o4_band(0.70, 0.80) == "A", "band: exactly prior-0.10 -> A (>=)")

    # wilcoxon guard
    check(wilcoxon_p(np.zeros(10)) == 1.0, "wilcoxon: all-zero delta -> p=1")
    check(np.isnan(wilcoxon_p(np.array([]))), "wilcoxon: empty -> nan")

    # stability/host calibers on crafted matrices with known answers
    idx = ["a", "b", "c"]
    M1 = pd.DataFrame([[0, 1, 2], [1, 0, 3], [2, 3, 0]], index=idx,
                      columns=idx, dtype=float)
    M2 = M1 * 2.0                        # rank-identical
    check(abs(_stability_mean({1: M1, 2: M2}) - 1.0) < 1e-12,
          "stability: rank-identical matrices -> mean pairwise rho = 1")
    A1 = M1.copy()
    A2 = -M1                             # reversed ranks
    got = _host_band_pairs({1: A1, 2: A2})[0]["spearman"]
    check(abs(got - (-1.0)) < 1e-9, "host band: reversed matrices -> rho = -1")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["verdict", "observables-15seed",
                             "mimic-np-descriptive", "selftest"])
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set before the "
              "interpreter starts (finding B48).", file=sys.stderr)
        return 2
    if a.stage == "selftest":
        return _selftest()
    if a.stage == "observables-15seed":
        return stage_observables_15seed()
    if a.stage == "mimic-np-descriptive":
        return stage_mimic_np_descriptive()
    return stage_verdict()


if __name__ == "__main__":
    raise SystemExit(main())
