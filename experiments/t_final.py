"""t_final.json -- the single evidence source (P5R-F SS1.1, second-review
P0-A). Aggregates every statistic the paper's abstract, Tables 4/6/7/8,
Fig. 3 and the ESM statistical tables consume, under the unified estimand
(P5R-F SS1.3):

  * primary statistic T = mean of seed-level medians;
  * exact test = exact enumeration under seed-block sign-exchangeability
    (all 2^n joint block sign flips);
  * 95% CI for T = seed-only bootstrap (targets/regimes are the fixed
    reference set; inference scope = training randomness);
  * pooled median = SECONDARY robust summary only.

Also performs the SS2 recovery re-inference (zero training): every
recovery comparison re-inferred with the 3 regimes as within-seed blocks
and the 5 synthetic seeds as independent units, from the stored per-cell
scores. Leakage is reported per batch (original / confirmatory) with the
pooled gen-3 row as secondary. Consumers must read THIS file; the
consumer selftests assert bitwise agreement with it.

    env PYTHONHASHSEED=2025 python experiments/t_final.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

R = CODE_ROOT / "results"
OUT = R / "T5_stats" / "t_final.json"

ESTIMAND = {
    "primary_T": "mean of seed-level medians",
    "exact_test": "exact enumeration under seed-block "
                  "sign-exchangeability (all 2^n joint block sign flips)",
    "ci_T": "seed-only percentile bootstrap, 10,000 resamples, rng seed "
            "20260831; targets/regimes are the fixed reference set -- "
            "the inference scope is training randomness",
    "secondary": "pooled median (robust summary; no test attached)",
}


def _symmetric_host_band() -> dict:
    """The host band under information symmetry (T6.1, addendum 2026-08-29b)."""
    f = CODE_ROOT / "results" / "T6_symmetry" / "no_oracle_band.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing: the scoreboard would then print the host band "
            f"measured under the privileged error signal, which is not the "
            f"band the Discussion compares against.")
    b = json.loads(f.read_text())["datasets"]
    return {ds: float(b[ds]["band_noOracle"]["mean"]) for ds in ("MIMIC", "eICU")}


def _family(blocks: dict, holm_p: float | None = None) -> dict:
    from t51_cluster_stats import (aggregates, seed_boot_ci_T,
                                   sign_flip_exact)
    m1 = sign_flip_exact(blocks)
    ag = aggregates(blocks)
    neg = sum(1 for v in ag["seed_medians"].values() if v < 0)
    out = {"n_seeds": m1["n_blocks"],
           "T": round(m1["observed_stat_mean_of_block_medians"], 6),
           "ci95_T": [round(x, 6) for x in seed_boot_ci_T(blocks)],
           "p_exact": m1["p_two_sided"],
           "floor": m1["floor_two_sided"],
           "seeds_negative": f"{neg}/{m1['n_blocks']}",
           "pooled_median_secondary": round(m1["effect_pooled_median"], 6)}
    if holm_p is not None:
        out["p_holm"] = holm_p
    return out


def build() -> dict:
    from t51_cluster_stats import (EXP_SEEDS, SEEDS, fam_faith_expanded,
                                   holm)
    from t43_verdict import _row_rhos

    # ---- faithfulness (both tables, 15 seeds), unified estimand -------- #
    fai = {}
    blocks_by_ds = {ds: fam_faith_expanded(ds, "SNI")
                    for ds in ("MIMIC", "eICU")}
    from t51_cluster_stats import sign_flip_exact
    raw_p = {ds: sign_flip_exact(blocks_by_ds[ds])["p_two_sided"]
             for ds in ("MIMIC", "eICU")}
    hp = holm([raw_p["MIMIC"], raw_p["eICU"]])
    for ds, h in zip(("MIMIC", "eICU"), hp):
        fai[ds] = _family(blocks_by_ds[ds], holm_p=h)

    # per-object 15-seed rho medians for Table 6 (SNI-D and TAP only; the
    # MF readouts stay a five-seed descriptive subset by design)
    rho_med = {}
    for ds in ("MIMIC", "eICU"):
        rd_all, rp_all = [], []
        P = None
        for s in SEEDS + EXP_SEEDS:
            A = pd.read_csv(R / "T3_faithfulness"
                            / f"A_{ds}_seed{s}_cpu_t2.csv", index_col=0)
            D = pd.read_csv(R / "T3_faithfulness"
                            / f"D_retrained_{ds}_seed{s}_cpu_t2.csv",
                            index_col=0)
            if P is None:
                feats = list(D.index)
                P = pd.read_csv(R / "T2g_prior_attribution"
                                / f"P_{ds}_seed1_cpu_t2.csv", index_col=0
                                ).reindex(index=feats, columns=feats)
            feats_l = list(D.index)
            for f in A.index:
                srcs = [c for c in feats_l if c != f]
                a = A.loc[f, srcs].astype(float)
                top_a = set(a.nlargest(3).index)
                for tgt_list, M in ((rd_all, D), (rp_all, P)):
                    m = M.loc[f, srcs].astype(float)
                    from scipy.stats import spearmanr
                    tgt_list.append(
                        (float(spearmanr(m, a).statistic),
                         len(set(m.nlargest(3).index) & top_a) / 3.0))
        rho_med[ds] = {
            "SNI-D": {"rho_median":
                          round(float(np.median([x[0] for x in rd_all])), 4),
                      "top3_mean":
                          round(float(np.mean([x[1] for x in rd_all])), 4)},
            "TAP": {"rho_median":
                        round(float(np.median([x[0] for x in rp_all])), 4),
                    "top3_mean":
                        round(float(np.mean([x[1] for x in rp_all])), 4)},
            "n_cells": len(rd_all)}

    # ---- no-prior (both tables, 15 seeds) ------------------------------ #
    # P5R-H SS2 / third review SS5.5: the two no-prior tables answer ONE
    # comparative claim, so they form a single two-test family under Holm --
    # not two single-comparison families. Both raw p values sit at the
    # enumeration floor, so the adjusted decision is unchanged; the point is
    # to avoid a family definition that looks like selective splitting.
    np_blocks = {ds: fam_faith_expanded(ds, "NoPrior")
                 for ds in ("MIMIC", "eICU")}
    np_raw = {ds: sign_flip_exact(np_blocks[ds])["p_two_sided"]
              for ds in ("MIMIC", "eICU")}
    np_holm = holm([np_raw["MIMIC"], np_raw["eICU"]])
    npf = {ds: _family(np_blocks[ds], holm_p=h)
           for ds, h in zip(("MIMIC", "eICU"), np_holm)}

    # ---- recovery re-inference (SS2): regimes within seed blocks ------- #
    six = pd.read_csv(R / "T4_perm_on_sni" / "t4f_sixway_cells.csv")

    def rec_pair(m_a: str, m_b: str) -> dict:
        pa = six[six.method == m_a].set_index(["seed", "regime"]).auroc
        pb = six[six.method == m_b].set_index(["seed", "regime"]).auroc
        d = (pa - pb).unstack("regime")
        assert d.shape == (5, 3) and not d.isna().any().any(), \
            f"recovery pair {m_a} vs {m_b}: incomplete grid"
        return _family({s: d.loc[s].to_numpy() for s in d.index})

    # D vs TAP from the pilot machinery (already seed-block in t51)
    t51 = json.loads((R / "T5_stats" / "t51_cluster_stats.json"
                      ).read_text())["families"]
    rec_d_tap = t51["recovery|pilot"]
    from t51_cluster_stats import fam_recovery, seed_boot_ci_T
    blocks_rec = fam_recovery(False)
    # ---- the same pair under information symmetry (T61 addendum d SS2) --- #
    # Same estimator as rec_pair, different cells: one training run per cell
    # emitted D AND both probes, so the host draw cancels INSIDE the pair.
    # Recomputed here from the cells rather than copied from the run's own
    # JSON, so two independent implementations have to agree (asserted below).
    def fair_pair(m_a: str, m_b: str) -> dict:
        c = pd.read_csv(R / "T6_symmetry" /
                        "fair_same_host_recovery_cells.csv")
        pa = c[c.method == m_a].set_index(["seed", "regime"]).auroc
        pb = c[c.method == m_b].set_index(["seed", "regime"]).auroc
        d = (pa - pb).unstack("regime")
        assert d.shape == (5, 3) and not d.isna().any().any(), \
            f"fair pair {m_a} vs {m_b}: incomplete grid"
        return _family({s_: d.loc[s_].to_numpy() for s_ in d.index})

    sym = fair_pair("Permutation-on-SNI-fair-noOracle", "SNI-D-fairhost")
    orc = fair_pair("Permutation-on-SNI-fair-oracle", "SNI-D-fairhost")
    old_pair = rec_pair("Permutation-on-SNI", "SNI-D-retrained")
    # The superseded reading is KEPT, and marked in place. Deleting it would
    # make the manuscript's history unreadable; overwriting it would make the
    # change invisible. It stays, with what replaced it and by how much.
    old_pair = dict(old_pair, superseded_by="probe_vs_D_same_host_symmetric",
                    superseded_on="2026-08-29",
                    superseded_because=(
                        "oracle caliber -- the probe's error signal was taken "
                        "from values withheld from the imputer -- and paired "
                        "with the T4F-RETRAINED host's D rather than the "
                        "host the probe itself ran on"),
                    delta_T_symmetric_minus_superseded=round(
                        sym["T"] - old_pair["T"], 6))
    recovery = {
        "D_vs_TAP": _family(blocks_rec),
        "probe_vs_D_same_host_symmetric": sym,
        "probe_vs_D_same_host_oracle_control": orc,
        "oracle_contribution_T": round(orc["T"] - sym["T"], 6),
        "probe_vs_D_retrained": old_pair,
        "scope": "five synthetic seeds as independent units; three "
                 "regimes as within-seed blocks; one graph family -- no "
                 "recovery comparison reaches the 0.0625 two-sided "
                 "enumeration floor's significance threshold by "
                 "construction",
        "consistency_check_D_vs_TAP_p":
            {"here": None, "t51": rec_d_tap["m1"]["p_two_sided"]},
    }
    recovery["consistency_check_D_vs_TAP_p"]["here"] = \
        recovery["D_vs_TAP"]["p_exact"]
    assert (recovery["D_vs_TAP"]["p_exact"]
            == rec_d_tap["m1"]["p_two_sided"]), "recovery recompute drift"

    # Two implementations, one number: this file recomputes the fair pair from
    # the cells; experiments/fair_same_host_recovery.py computed it from the
    # matrices as it wrote them. If they ever disagree, one of them is wrong
    # and neither may be printed.
    _fair = json.loads((R / "T6_symmetry" /
                        "fair_same_host_recovery.json").read_text())
    for _k, _blk in (("probe_vs_D_same_host_symmetric",
                      "probe_vs_D_same_host_no_oracle"),
                     ("probe_vs_D_same_host_oracle_control",
                      "probe_vs_D_same_host_oracle_control")):
        assert abs(recovery[_k]["T"]
                   - _fair[_blk]["T_mean_of_seed_medians"]) < 5e-7, (
            f"{_k}: t_final {recovery[_k]['T']} vs the run's own "
            f"{_fair[_blk]['T_mean_of_seed_medians']}")
        assert (recovery[_k]["p_exact"]
                == _fair[_blk]["exact_sign_enumeration"]["p_two_sided"]), \
            f"{_k}: p disagrees between the two implementations"
    assert (recovery["probe_vs_D_retrained"]["T"]
            == _fair["archived_superseded"]["T"]), \
        "the superseded number in t_final and in the fair run disagree"

    # ---- no-prior recovery (5 synth seeds, secondary interval) --------- #
    np_rec = t51["noprior_recovery|pilot"]
    from t51_cluster_stats import fam_recovery as _fr
    npr_blocks = _fr(True)
    noprior_recovery = _family(npr_blocks)

    # ---- leakage: per batch primary, pooled secondary ------------------ #
    det = pd.read_csv(R / "T4_leakage" / "t42_detection.csv")

    def _mcnemar_orig(a: str, b: str) -> dict:
        from math import comb
        g = det[(det.condition == "interaction") & (det.kind == "inj")]
        pa = g[g.object == a].set_index(["dataset", "host_seed"]).detected
        pb = g[g.object == b].set_index(["dataset", "host_seed"]).detected
        pa, pb = pa.sort_index(), pb.sort_index()
        x = int(((pa) & (~pb)).sum())
        y = int(((~pa) & (pb)).sum())
        n = x + y
        pv = 1.0 if n == 0 else min(1.0, 2.0 * sum(
            comb(n, i) for i in range(min(x, y) + 1)) * 0.5 ** n)
        return {"batch": "original", "pair": f"{a} vs {b}",
                "a_only": x, "b_only": y, "n_pairs": int(len(pa)),
                "p_two_sided": round(pv, 6),
                "holm_two_test_family_floor": 0.0625,
                "note": "post-hoc exploratory paired contrast: the original committed rules (6535787) fixed per-class reporting and the interaction win-threshold, designating no single primary inferential contrast (rule archaeology, P5R-H SS1.1)"}

    t42 = json.loads((R / "T4_leakage" / "t42_summary.json").read_text())
    conf = json.loads((R / "T4_leakage" / "t42_confirmatory.json"
                       ).read_text())
    nb = t42["null_exact_binomial"]
    leakage = {
        "batches": {
            "original": {
                "interaction": t42["interaction_counts"],
                "discrepancy_control": t42["decoy_false_positives"],
                "null_per_object": {o: f"{v['detected']}/{v['n']}"
                                    for o, v in nb.items()},
                "null_rate_per_object": t42["null_detection_rate"],
            },
            "confirmatory": {
                "interaction": {o: conf["counts"][o]["inj_confirmatory"]
                                ["detected"]
                                for o in conf["counts"]},
                "null_per_object": {
                    o: f"{conf['counts'][o]['null_confirmatory']['detected']}"
                       f"/{conf['counts'][o]['null_confirmatory']['n']}"
                    for o in conf["counts"]},
            },
        },
        "pooled_gen3_secondary": {o: f"{conf['counts'][o]['inj_pooled_gen3']['detected']}"
                                     f"/{conf['counts'][o]['inj_pooled_gen3']['n']}"
                                  for o in conf["counts"]},
        "probe_null_by_batch": {"original": nb["Permutation-on-SNI"],
                                "confirmatory": {
            "detected": conf["counts"]["Permutation-on-SNI"]
                        ["null_confirmatory"]["detected"],
            "n": conf["counts"]["Permutation-on-SNI"]
                 ["null_confirmatory"]["n"]}},
        # P5R-K SS2 (fourth review SS4.2): the (6/42)^6 bound is withdrawn.
        # It treated a rate estimated from 42 draws as a known probability
        # and the six detections as i.i.d.; the observed rates stay.
        "probe_interaction_counts":
            t42["probe_interaction_chance_bound"]["interaction_detected"],
        "mcnemar_confirmatory": conf["mcnemar_D_vs_TAP"],
        "mcnemar_D_vs_probe_original": _mcnemar_orig(
            "SNI-D", "Permutation-on-SNI"),
        "calibration": "thresholds calibrated on >=20 random-proxy "
                       "retrainings per (object, dataset, target row); "
                       "observed null rates reported per object and "
                       "batch",
    }

    # ---- probe-2: repeat-draw replication + group-permutation ---------- #
    p2 = json.loads((R / "T5_probe2" / "probe2_summary.json").read_text())
    p2q = json.loads((R / "T5_probe2" / "probe2_qualifiers.json"
                      ).read_text())
    probe2 = {"role": "repeat-draw replication (tau=0.8, all singletons) "
                      "+ group-permutation sensitivity (tau=0.6/0.7); "
                      "NOT an independent second readout",
              "families": p2["families"], "qualifiers": p2q}

    red = json.loads((R / "T5_stats" / "t51_redundancy_sensitivity.json"
                      ).read_text())

    # ---- branches + scoreboard descriptive block (P5R-H SS3 / P1-5) ---- #
    # Fig 3 is generated from THIS file alone; branch letters come from the
    # committed t51 machinery, the descriptive cells from their artifacts.
    t51x_fam = json.loads((R / "T5_stats" / "t51_expanded_15seed.json"
                           ).read_text())["families"]
    t51_fam = json.loads((R / "T5_stats" / "t51_cluster_stats.json"
                          ).read_text())["families"]
    for ds in ("MIMIC", "eICU"):
        fai[ds]["branch"] = t51x_fam[f"faithfulness|{ds}"]["branch"]
        npf[ds]["branch"] = t51x_fam[f"noprior_faithfulness|{ds}"]["branch"]
    recovery["branch"] = t51_fam["recovery|pilot"]["branch"]

    import re as _re
    fw = json.loads((R / "T3_five_way" / "fiveway_summary.json").read_text())
    fs = json.loads((R / "T3_faithfulness" / "faithfulness_summary.json"
                     ).read_text())
    host = pd.read_csv(R / "T4_perm_on_sni"
                       / "perm_on_sni_real_stability.csv")
    pilot = pd.read_csv(R / "T2.5_pilot" / "pilot_full_cells.csv")
    macros = (CODE_ROOT / "reporting" / "out" / "cost_macros.tex").read_text()
    a3 = (CODE_ROOT / "reporting" / "out" / "a3_macros.tex").read_text()

    def _macro(text, name):
        m = _re.search(r"\\newcommand\{\\" + name
                       + r"\}\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        assert m, f"macro {name} missing"
        return m.group(1)

    scoreboard_desc = {
        "stability_rows12_mean": {
            o: {ds: fw[ds][o]["stability_rows12_mean"]
                for ds in ("MIMIC", "eICU")}
            for o in ("SNI-D", "MissForest-importance",
                      "SHAP-on-MissForest", "Permutation-on-MissForest")},
        "faith_rho_mean": {
            o: {ds: fs[ds][o]["rho_mean"] for ds in ("MIMIC", "eICU")}
            for o in ("MissForest-importance", "SHAP-on-MissForest",
                      "Permutation-on-MissForest")},
        "host_band_mean": {ds: float(host[host.dataset == ds
                                          ].spearman.mean())
                           for ds in ("MIMIC", "eICU")},
        # T6.1: the same band with the probe's error signal taken from the
        # host's own completed table instead of the withheld values. Both are
        # carried: the archived one is what the submitted version reported,
        # the symmetric one is what the Discussion compares D against.
        "host_band_mean_symmetric": _symmetric_host_band(),
        "pilot_auroc": pilot.groupby("method").auroc.mean().round(6
                                                                  ).to_dict(),
        # The retired T4F label used to be carried here, renamed so it could
        # not read as current. Tenth review P0-1: renaming is not removing,
        # and the canonical fact store is exactly where it must not be -- two
        # byte-identical copies of t_final.json shipped with
        # SAME_HOST_POSTHOC_WINS inside them while the response letter said
        # the current fact store held no such thing. It is gone from here.
        # The label, when it was retired and why, and the digests of the
        # frozen artifacts that carry it, are in evidence/AUDIT_HISTORY.json,
        # which is marked HISTORICAL - NOT CANONICAL and which no generator
        # reads.
        "cost_display": {
            "grid_range": _macro(macros, "costRatioRange"),
            "single_thread_range": _macro(a3, "costRatioSTRange"),
        },
    }

    return {"built_from": {
                "t51_expanded": "results/T5_stats/t51_expanded_15seed.json",
                "sixway": "results/T4_perm_on_sni/t4f_sixway_cells.csv",
                "fair_same_host": "results/T6_symmetry/"
                                  "fair_same_host_recovery_cells.csv",
                "t42": "results/T4_leakage/t42_summary.json",
                "confirmatory": "results/T4_leakage/t42_confirmatory.json",
                "probe2": "results/T5_probe2/*.json",
                "scoreboard_desc": "fiveway_summary + faithfulness_summary + perm_on_sni_real_stability + pilot_full_cells + cost/a3 macros (the T4F verdict is carried as retired audit history and is not an input)"},
            "estimand": ESTIMAND,
            "faithfulness": fai,
            "faithfulness_rho_medians_15seed": rho_med,
            "noprior_faithfulness": npf,
            "recovery": recovery,
            "noprior_recovery": noprior_recovery,
            "leakage": leakage,
            "probe2_sensitivity": probe2,
            "scoreboard_desc": scoreboard_desc,
            "redundancy_sensitivity": red["cells"]}


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    from t51_cluster_stats import seed_boot_ci_T, sign_flip_exact
    blocks = {s: [-(1 + 0.01 * s + 0.001 * j) for j in range(12)]
              for s in range(5)}
    f = _family(blocks)
    m1 = sign_flip_exact(blocks)
    check(f["T"] == round(m1["observed_stat_mean_of_block_medians"], 6)
          and f["p_exact"] == m1["p_two_sided"]
          and f["seeds_negative"] == "5/5",
          "_family: T, p, negative-seed count consistent with machinery")
    check(f["ci95_T"][0] <= f["T"] <= f["ci95_T"][1],
          "_family: CI brackets T")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        print("REFUSING TO RUN: PYTHONHASHSEED must be set.",
              file=sys.stderr)
        return 2
    if a.selftest:
        return _selftest()
    out = build()
    OUT.write_text(json.dumps(out, indent=1))
    for ds in ("MIMIC", "eICU"):
        f = out["faithfulness"][ds]
        print(f"faith {ds}: T={f['T']:+.4f} CI{f['ci95_T']} "
              f"p={f['p_exact']:.6f} holm={f['p_holm']:.6f} "
              f"neg={f['seeds_negative']}")
    r = out["recovery"]["probe_vs_D_same_host_symmetric"]
    o = out["recovery"]["probe_vs_D_retrained"]
    print(f"recovery probe-vs-D SUPERSEDED (oracle caliber, cross-host): "
          f"T={o['T']:+.4f}")
    print(f"recovery probe-vs-D: T={r['T']:+.4f} CI{r['ci95_T']} "
          f"p={r['p_exact']:.4f} (floor {r['floor']})")
    print(f"[ok] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
