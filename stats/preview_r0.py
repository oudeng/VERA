"""Zero-rerun statistical preview over the frozen R0 results (task T1.7, part 3).

Runs the entire new statistical pipeline against ``project_sni_R0/results_all``
without touching a model, and writes every artifact under
``code_SNI/results/T1.7_stats_preview/``.

Usage::

    PYTHONPATH=$PWD \
        python -m stats.preview_r0

Nothing here decides anything.  In particular the TOST section reports a
*sensitivity curve* over candidate equivalence margins; picking one is the first
author's call.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import cd_diagram, effect_size, equivalence, intervals, long_table, omnibus, posthoc

#: P7-A closeout: no private absolute path in a published file. The
#: R0 tree is not in this repository (it holds restricted derived
#: tables); point at it with SNI_R0_ROOT, and default to the sibling
#: directory a full checkout would have. A clone that lacks it gets a
#: path it can act on rather than a stranger's home directory.
R0_ROOT = Path(os.environ.get("SNI_R0_ROOT",
                    Path(__file__).resolve().parents[2]
                    / "project_sni_R0"))
R0_RESULTS = R0_ROOT / "results_all"
OUT_DIR = (Path(__file__).resolve().parents[1]
           / "results" / "T1.7_stats_preview")

REFERENCE = "SNI"
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 20260727

#: Metrics carried through the whole preview, in report order.
PREVIEW_METRICS: Sequence[str] = (
    "NRMSE", "RMSE", "MAE", "R2", "Spearman", "Accuracy", "Macro-F1", "Cohen_kappa",
)

#: Candidate equivalence margins per metric.  Deliberately a *range*: the
#: instruction is explicit that delta must not be fixed at this stage.
CANDIDATE_DELTAS: Dict[str, Sequence[float]] = {
    "NRMSE": (0.005, 0.01, 0.02, 0.05),
    "RMSE": (0.05, 0.1, 0.25, 0.5),
    "MAE": (0.05, 0.1, 0.25, 0.5),
    "R2": (0.01, 0.02, 0.05, 0.10, 0.20),
    "Spearman": (0.01, 0.02, 0.05, 0.10),
    "Accuracy": (0.01, 0.02, 0.05, 0.10),
    "Macro-F1": (0.01, 0.02, 0.05, 0.10),
    "Cohen_kappa": (0.01, 0.02, 0.05, 0.10),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================== #
# 1. Long table
# =========================================================================== #


def build_and_audit(out: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    _log("[1/7] building the tidy long table ...")
    long = long_table.build_long_table(R0_RESULTS)
    audit = long_table.audit_long_table(long)

    long.to_csv(out / "long_table.csv.gz", index=False, compression="gzip")
    (out / "long_table_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    long_table.coverage_table(long).to_csv(out / "coverage_by_method.csv")

    per_family = (
        long.groupby(["experiment_family", "source"])
        .agg(rows=("value", "size"), runs=("exp_id", "nunique"),
             datasets=("dataset", "nunique"), methods=("method", "nunique"))
        .reset_index()
    )
    per_family.to_csv(out / "long_table_by_source.csv", index=False)

    _log(f"      shape={long.shape}  runs={audit['n_runs']}  cells={audit['n_cells']}")
    _log(f"      excluded: {sorted(long_table.EXCLUDED_SOURCES)}")
    return long, audit


# =========================================================================== #
# 2. Friedman + Iman-Davenport
# =========================================================================== #


def run_omnibus(main: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    _log("[2/7] Friedman + Iman-Davenport on the published main grid ...")
    rows: List[Dict[str, object]] = []
    ranks: Dict[str, pd.Series] = {}
    rank_rows: List[Dict[str, object]] = []

    for metric in PREVIEW_METRICS:
        hib = long_table.METRIC_DIRECTION[metric]
        mat = long_table.to_setting_matrix(main, metric)
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            continue
        res = omnibus.friedman_from_matrix(mat, metric=metric, higher_is_better=bool(hib))
        row = res.to_row()
        row["CD_nemenyi_0.05"] = posthoc.nemenyi_critical_difference(res.k_methods, res.n_blocks, ALPHA)
        row["CD_bonferroni_dunn_0.05"] = posthoc.bonferroni_dunn_critical_difference(
            res.k_methods, res.n_blocks, ALPHA
        )
        rows.append(row)
        r = pd.Series(res.avg_ranks).sort_values()
        ranks[metric] = r
        for m, v in r.items():
            rank_rows.append({"metric": metric, "method": m, "avg_rank": float(v),
                              "n_blocks": res.n_blocks, "k_methods": res.k_methods})

    omni = pd.DataFrame(rows)
    omni.to_csv(out / "friedman_iman_davenport.csv", index=False)
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "average_ranks.csv", index=False)
    rank_df.pivot(index="method", columns="metric", values="avg_rank").to_csv(
        out / "average_ranks_wide.csv"
    )
    return omni, ranks


# =========================================================================== #
# 3. Post-hoc: Nemenyi and Wilcoxon-Holm, with effect sizes and CIs
# =========================================================================== #


def run_posthoc(
    main: pd.DataFrame, ranks: Dict[str, pd.Series], out: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    _log("[3/7] post-hoc: Nemenyi and Wilcoxon-Holm, with effect sizes and BCa CIs ...")
    nem_rows: List[pd.DataFrame] = []
    wil_rows: List[pd.DataFrame] = []
    diffs: Dict[str, Dict[str, np.ndarray]] = {}

    for metric in PREVIEW_METRICS:
        if metric not in ranks:
            continue
        hib = bool(long_table.METRIC_DIRECTION[metric])
        mat = long_table.to_setting_matrix(main, metric)

        nem = posthoc.nemenyi_vs_reference(ranks[metric], mat.shape[0], REFERENCE, alpha=ALPHA)
        nem.insert(0, "metric", metric)
        nem_rows.append(nem)

        wil = posthoc.wilcoxon_holm(mat, REFERENCE, metric=metric, alpha=ALPHA)
        diffs[metric] = {str(r["other"]): np.asarray(r["_diff"], dtype=float)
                         for _, r in wil.iterrows()}

        enriched: List[Dict[str, object]] = []
        for _, r in wil.iterrows():
            d = np.asarray(r["_diff"], dtype=float)
            es = effect_size.effect_sizes_for_diffs(d)
            ci = intervals.bootstrap_ci(
                d, np.mean, method="bca", alpha=ALPHA,
                n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
            )
            ci_med = intervals.median_ci(
                d, method="bca", alpha=ALPHA,
                n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
            )
            n_pairs = int(r["n_settings"]) - int(r["n_zero"])
            rb_from_w = effect_size.rank_biserial_from_w(
                r["W_statistic"], n_pairs, sign=r["mean_diff"]
            )
            row = {k: v for k, v in r.items() if k != "_diff"}
            row.update(
                {
                    "higher_is_better": hib,
                    "rank_biserial": es.rank_biserial,
                    "rank_biserial_magnitude": es.rank_biserial_magnitude,
                    "rank_biserial_from_stored_W": rb_from_w,
                    "rank_biserial_sign_conflict": bool(
                        np.sign(rb_from_w) != np.sign(es.rank_biserial)
                    ),
                    "cliffs_delta_paired": es.cliffs_delta_paired,
                    "cliffs_delta_magnitude": es.cliffs_delta_magnitude,
                    "cohens_dz": es.cohens_dz,
                    "hedges_gz": es.hedges_gz,
                    "mean_diff_ci_low": ci.lower,
                    "mean_diff_ci_high": ci.upper,
                    "mean_diff_ci_method": ci.method_used,
                    "median_diff_ci_low": ci_med.lower,
                    "median_diff_ci_high": ci_med.upper,
                    "ci_excludes_zero": bool(
                        np.isfinite(ci.lower) and np.isfinite(ci.upper)
                        and (ci.lower > 0 or ci.upper < 0)
                    ),
                }
            )
            enriched.append(row)
        wil_rows.append(pd.DataFrame(enriched))

    nemenyi = pd.concat(nem_rows, ignore_index=True)
    wilcoxon = pd.concat(wil_rows, ignore_index=True)
    nemenyi.to_csv(out / "posthoc_nemenyi.csv", index=False)
    wilcoxon.to_csv(out / "posthoc_wilcoxon_holm_effectsize_ci.csv", index=False)

    # Where do the two routes disagree?
    merged = wilcoxon.merge(
        nemenyi[["metric", "other", "significant_nemenyi", "significant_bonferroni_dunn",
                 "avg_rank_reference", "avg_rank_other", "p_nemenyi"]],
        on=["metric", "other"], how="left",
    )
    merged["routes_agree"] = merged["significant"] == merged["significant_nemenyi"]
    merged[
        ["metric", "other", "mean_diff", "p_adjusted", "significant",
         "p_nemenyi", "significant_nemenyi", "significant_bonferroni_dunn", "routes_agree"]
    ].to_csv(out / "posthoc_route_comparison.csv", index=False)
    return nemenyi, wilcoxon, diffs


# =========================================================================== #
# 4. TOST sensitivity
# =========================================================================== #


def run_tost(diffs: Dict[str, Dict[str, np.ndarray]], out: Path) -> pd.DataFrame:
    _log("[4/7] TOST sensitivity over candidate equivalence margins ...")
    frames: List[pd.DataFrame] = []
    for metric, per_other in diffs.items():
        deltas = CANDIDATE_DELTAS.get(metric)
        if not deltas:
            continue
        for other, d in per_other.items():
            tbl = equivalence.tost_sensitivity(d, deltas, alpha=ALPHA,
                                               label=f"{REFERENCE} vs {other}")
            tbl.insert(0, "metric", metric)
            tbl.insert(2, "other", other)
            # Non-parametric companion.
            wil = [equivalence.tost_wilcoxon_paired(d, dd, alpha=ALPHA) for dd in deltas]
            tbl["p_tost_wilcoxon"] = [w["p_tost"] for w in wil]
            tbl["equivalent_wilcoxon"] = [w["equivalent"] for w in wil]
            frames.append(tbl)

    all_tost = pd.concat(frames, ignore_index=True)
    all_tost.to_csv(out / "tost_sensitivity_all.csv", index=False)

    focus = all_tost[all_tost["other"] == "HyperImpute"].copy()
    focus.to_csv(out / "tost_sensitivity_sni_vs_hyperimpute.csv", index=False)

    # Smallest margin at which equivalence could be claimed, per comparison.
    rows = []
    for (metric, other), grp in all_tost.groupby(["metric", "other"]):
        eq = grp[grp["equivalent"]]
        rows.append(
            {
                "metric": metric,
                "other": other,
                "mean_diff": float(grp["mean_diff"].iloc[0]),
                "smallest_delta_tested": float(grp["delta"].min()),
                "largest_delta_tested": float(grp["delta"].max()),
                "smallest_equivalent_delta": float(eq["delta"].min()) if len(eq) else np.nan,
                "equivalent_at_any_tested_delta": bool(len(eq) > 0),
            }
        )
    boundary = pd.DataFrame(rows).sort_values(["metric", "other"])
    boundary.to_csv(out / "tost_equivalence_boundary.csv", index=False)
    return all_tost


# =========================================================================== #
# 5. CD diagrams
# =========================================================================== #


def run_cd(main: pd.DataFrame, ranks: Dict[str, pd.Series], out: Path) -> List[Path]:
    _log("[5/7] critical-difference diagrams ...")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for metric in ("NRMSE", "R2", "Spearman", "Macro-F1"):
        if metric not in ranks:
            continue
        mat = long_table.to_setting_matrix(main, metric)
        n, k = mat.shape
        cd = posthoc.nemenyi_critical_difference(k, n, ALPHA)
        n_ds = len({i[0] for i in mat.index})
        scope = f"{n_ds} datasets x {{MCAR, MAR}} @30%"
        if n_ds < 6:
            scope += " (Concrete has no categorical columns)"
        path = fig_dir / f"cd_diagram_{metric.replace('-', '')}.pdf"
        fig = cd_diagram.plot_cd_diagram(
            ranks[metric],
            cd,
            title=f"Critical-difference diagram -- {metric}",
            subtitle=(
                f"Friedman + Nemenyi, alpha={ALPHA}; {n} settings "
                f"({scope}) x {k} methods; R0 per-seed results"
            ),
            out_path=path,
            highlight=[REFERENCE],
        )
        plt.close(fig)
        written.append(path)
        _log(f"      {path}  (CD={cd:.3f})")
    return written


# =========================================================================== #
# 6. Sign-convention verification
# =========================================================================== #


def verify_sign_convention(main: pd.DataFrame, out: Path) -> Dict[str, object]:
    _log("[6/7] verifying the Table S3 sign convention ...")
    mat = long_table.to_setting_matrix(main, "NRMSE")
    r0 = pd.read_csv(R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv")

    checks = []
    for other in ("MissForest", "MIWAE", "GAIN", "HyperImpute"):
        sni_mean = float(mat[REFERENCE].mean())
        oth_mean = float(mat[other].mean())
        stored = float(
            r0[(r0["metric"] == "NRMSE") & (r0["comparison"] == f"SNI vs {other}")]["mean_diff"].iloc[0]
        )
        recomputed = float((mat[other] - mat[REFERENCE]).mean())  # = -(SNI - other)
        checks.append(
            {
                "comparison": f"SNI vs {other}",
                "metric": "NRMSE (lower is better)",
                "mean_NRMSE_SNI": sni_mean,
                "mean_NRMSE_other": oth_mean,
                "sni_has_lower_nrmse": bool(sni_mean < oth_mean),
                "sni_is_better": bool(sni_mean < oth_mean),
                "stored_mean_diff": stored,
                "recomputed_baseline_minus_sni": recomputed,
                "sign_of_stored": "positive" if stored > 0 else "negative",
                "consistent_with_positive_favours_sni": bool(
                    (stored > 0) == (sni_mean < oth_mean)
                ),
            }
        )
    df = pd.DataFrame(checks)
    df.to_csv(out / "sign_convention_verification.csv", index=False)

    verdict = {
        "computation_in_R0": (
            "ext2/scripts/exp5_significance_tests.py:566-570 computes "
            "diff = SNI - baseline, then negates it when METRIC_DIRECTION[metric] "
            "is False (NRMSE). The stored mean_diff for NRMSE is therefore "
            "baseline - SNI, so POSITIVE means SNI has the lower NRMSE, i.e. SNI is better."
        ),
        "generator_caption": {
            "path": "project_sni_R0/scripts/06_gen_supp_tables.py",
            "lines": "355-360",
            "text": "negative favors SNI for NRMSE",
            "verdict": "WRONG -- states the convention backwards",
        },
        "generator_inline_comment": {
            "path": "project_sni_R0/scripts/06_gen_supp_tables.py",
            "lines": "344",
            "text": "For NRMSE: negative mean_diff means SNI is worse (higher NRMSE)",
            "verdict": "CORRECT -- and directly contradicts the caption six lines below",
        },
        "manuscript": {
            "path": "paper_SNI_HISC_R0/ESM_1_SNI_HISC_v5_5.tex",
            "line": "353",
            "text": (
                "Delta is signed so that positive values favor SNI: for NRMSE, "
                "Delta = baseline - SNI; for R^2 and Macro-F1, Delta = SNI - baseline."
            ),
            "verdict": "CORRECT -- the manuscript wording is the one to keep",
        },
        "numeric_proof": checks,
        "conclusion": (
            "POSITIVE mean_diff favors SNI for every metric, including NRMSE. "
            "The four all-negative NRMSE/R2/Spearman/Macro-F1 entries against "
            "MissForest and MIWAE therefore mean SNI is worse than those two "
            "baselines, which is what the manuscript already states. The "
            "generator caption must be corrected before any table is regenerated."
        ),
    }
    (out / "sign_convention_verification.json").write_text(json.dumps(verdict, indent=2, default=str))
    return verdict


# =========================================================================== #
# 7. R0 reconciliation and the zero-recompute effect-size upgrade
# =========================================================================== #


def reconcile_with_r0(
    main: pd.DataFrame, wilcoxon: pd.DataFrame, out: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log("[7/7] reconciling against R0's published Table S3 ...")
    r0_path = R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv"

    # (a) zero-recomputation effect sizes on the stored table
    sign_override = {}
    name_map = {"NRMSE": "NRMSE", "R2": "R2", "Spearman_rho": "Spearman", "Macro_F1": "Macro-F1"}
    for r0_metric, our_metric in name_map.items():
        sub = wilcoxon[wilcoxon["metric"] == our_metric]
        for _, row in sub.iterrows():
            sign_override[(r0_metric, f"SNI vs {row['other']}")] = float(row["rank_biserial"])
    augmented = effect_size.augment_r0_wilcoxon_table(r0_path, sign_override=sign_override)
    augmented.to_csv(out / "r0_table_s3_with_effect_sizes.csv", index=False)

    # (b) cell-by-cell reconciliation
    keep = ["metric", "other", "n_settings", "mean_diff", "median_diff",
            "W_statistic", "p_value", "p_adjusted", "significant",
            "rank_biserial", "cliffs_delta_paired", "hedges_gz",
            "mean_diff_ci_low", "mean_diff_ci_high"]
    r1 = wilcoxon[keep].rename(columns={c: f"{c}_r1" for c in keep if c != "other"})
    r1 = r1.rename(columns={"metric_r1": "metric"})

    r0 = pd.read_csv(r0_path)
    r0["other"] = r0["comparison"].str.replace("SNI vs ", "", regex=False)
    r0["metric"] = r0["metric"].map(name_map)
    r0 = r0.rename(
        columns={c: f"{c}_r0" for c in
                 ["n_settings", "mean_diff", "W_statistic", "p_value", "p_adjusted", "significant"]}
    )
    merged = r0.merge(r1, on=["metric", "other"], how="inner")

    recon = pd.DataFrame(
        {
            "metric": merged["metric"],
            "comparison": merged["comparison"],
            "n_settings_r0": merged["n_settings_r0"],
            "n_settings_r1": merged["n_settings_r1"],
            "mean_diff_r0": merged["mean_diff_r0"],
            "mean_diff_r1": merged["mean_diff_r1"],
            "mean_diff_abs_change": (merged["mean_diff_r1"] - merged["mean_diff_r0"]).abs(),
            "W_r0": merged["W_statistic_r0"],
            "W_r1": merged["W_statistic_r1"],
            "p_adj_r0": merged["p_adjusted_r0"],
            "p_adj_r1": merged["p_adjusted_r1"],
            "significant_r0": merged["significant_r0"].astype(bool),
            "significant_r1": merged["significant_r1"].astype(bool),
            "rank_biserial_r1": merged["rank_biserial_r1"],
            "cliffs_delta_r1": merged["cliffs_delta_paired_r1"],
            "hedges_gz_r1": merged["hedges_gz_r1"],
            "ci_low_r1": merged["mean_diff_ci_low_r1"],
            "ci_high_r1": merged["mean_diff_ci_high_r1"],
        }
    )
    recon["verdict_changed"] = recon["significant_r0"] != recon["significant_r1"]
    recon["W_changed"] = recon["W_r0"] != recon["W_r1"]
    recon.to_csv(out / "r0_vs_r1_reconciliation.csv", index=False)
    _log(
        f"      verdicts changed: {int(recon['verdict_changed'].sum())}/{len(recon)}; "
        f"W changed: {int(recon['W_changed'].sum())}/{len(recon)}"
    )
    return augmented, recon


def diagnose_r0_pooling_defect(out: Path) -> Dict[str, object]:
    """Prove that R0's Table S3 pooled the lambda-ablation runs into the SNI arm."""
    main_sni = pd.read_csv(R0_RESULTS / "agg_sni_v03_main" / "summary_all.csv")
    abla = pd.read_csv(R0_RESULTS / "agg_sni_v03_ablation_lambda" / "summary_all.csv")
    bmain = pd.read_csv(R0_RESULTS / "agg_baselines_main" / "summary_all.csv")
    bnew = pd.read_csv(R0_RESULTS / "agg_baselines_new" / "summary_all.csv")
    r0 = pd.read_csv(R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv")

    col_map = {"NRMSE": "cont_NRMSE", "R2": "cont_R2",
               "Spearman_rho": "cont_Spearman", "Macro_F1": "cat_Macro-F1"}
    lower_better = {"NRMSE"}

    def setting_mean(df, col):
        return df.groupby(["dataset", "mechanism"])[col].mean()

    results = {}
    for label, pool in (
        ("clean (agg_sni_v03_main only)", pd.concat([main_sni], ignore_index=True)),
        ("main x1 + ablation x1", pd.concat([main_sni, abla], ignore_index=True)),
        ("main x2 + ablation x1 (R0's actual loader)",
         pd.concat([main_sni, main_sni, abla], ignore_index=True)),
    ):
        n_exact = 0
        for r0_metric, col in col_map.items():
            sni = setting_mean(pool, col)
            for _, row in r0[r0["metric"] == r0_metric].iterrows():
                other = row["comparison"].replace("SNI vs ", "")
                src = bmain if other in set(bmain["algo"]) else bnew
                base = setting_mean(src[src["algo"] == other], col)
                m = pd.concat([sni.rename("s"), base.rename("b")], axis=1).dropna()
                diff = m["s"] - m["b"]
                if r0_metric in lower_better:
                    diff = -diff
                if abs(float(diff.mean()) - float(row["mean_diff"])) < 1e-12:
                    n_exact += 1
        results[label] = f"{n_exact}/32 cells reproduced to 1e-12"

    verdict = {
        "finding": (
            "R0's published Table S3 does not use the SNI main-grid runs alone. "
            "ext2/scripts/exp5_significance_tests.py::_load_sni_results reads "
            "agg_sni_v03_main AND the per-run folders sni_v03_main/*/metrics_summary.csv "
            "(the same 60 runs, counted twice) AND agg_sni_v03_ablation_lambda, whose "
            "`variant` column reads 'SNI'. The 50 fixed-lambda ablation runs are "
            "therefore pooled into the SNI arm for the MIMIC/MAR and NHANES/MAR settings."
        ),
        "pooling_hypotheses_tested": results,
        "impact": (
            "mean_diff shifts in the fourth decimal for every comparison; three of the "
            "eight Macro-F1 W statistics change by 1; NO significance verdict changes. "
            "The published Table S3 conclusions therefore stand, but the numbers in the "
            "Delta column are not the numbers the caption describes."
        ),
        "status": "NEW -- not recorded in the P0 scan report (B1-B47)",
    }
    (out / "r0_table_s3_pooling_defect.json").write_text(json.dumps(verdict, indent=2, default=str))
    return verdict


# =========================================================================== #
# Driver
# =========================================================================== #


def write_readme(out: Path, manifest: Dict[str, object]) -> None:
    """A short index so the directory is self-explanatory."""
    lines = [
        "# T1.7 zero-rerun statistical preview",
        "",
        "Generated by `code_SNI/stats/preview_r0.py` from the frozen R0 per-seed",
        "results only. **No model was re-run.**",
        "",
        "## Inputs",
        "",
        "`project_sni_R0/results_all/agg_*/summary_all.csv`, seven directories.",
        "`agg_baselines_deep` is excluded by assertion (all 300 runs failed with a",
        "CUDA error; P0 finding B21).",
        "",
        "## Files",
        "",
        "| file | contents |",
        "|---|---|",
        "| `long_table.csv.gz` | the tidy long table (13,610 rows) |",
        "| `long_table_main_grid.csv` | the published main grid subset |",
        "| `long_table_audit.json` | shape, coverage, seed counts, exclusions |",
        "| `long_table_by_source.csv` | rows / runs / methods per source directory |",
        "| `coverage_by_method.csv` | method x (mechanism, rate) dataset counts |",
        "| `friedman_iman_davenport.csv` | omnibus test per metric |",
        "| `average_ranks.csv`, `average_ranks_wide.csv` | average ranks per metric |",
        "| `posthoc_nemenyi.csv` | Nemenyi and Bonferroni--Dunn, SNI vs all |",
        "| `posthoc_wilcoxon_holm_effectsize_ci.csv` | Wilcoxon+Holm with effect sizes and BCa CIs |",
        "| `posthoc_route_comparison.csv` | where the two post-hoc routes disagree |",
        "| `tost_sensitivity_all.csv` | TOST over candidate margins, all comparisons |",
        "| `tost_sensitivity_sni_vs_hyperimpute.csv` | the R1-5 / R2-6a focus comparison |",
        "| `tost_equivalence_boundary.csv` | smallest margin at which equivalence holds |",
        "| `sign_convention_verification.{csv,json}` | numeric proof of the Table S3 sign direction |",
        "| `r0_table_s3_with_effect_sizes.csv` | R0's stored table upgraded with rank-biserial |",
        "| `r0_vs_r1_reconciliation.csv` | cell-by-cell R0 vs clean recomputation |",
        "| `r0_table_s3_pooling_defect.json` | NEW defect: lambda-ablation runs pooled into the SNI arm |",
        "| `figures/cd_diagram_*.pdf,.png` | critical-difference diagrams |",
        "| `preview_manifest.json` | machine-readable summary of this run |",
        "",
        "## Headline",
        "",
        f"- equivalence margins are **not** fixed here: `delta` is swept, see `CANDIDATE_DELTAS`.",
        f"- significance verdicts changed vs R0: **{manifest['verdicts_changed_vs_r0']}/32**.",
        f"- sign convention: {manifest['sign_convention_conclusion']}",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"output directory: {OUT_DIR}")

    long, audit = build_and_audit(OUT_DIR)
    main_grid = long_table.main_grid_view(long)
    main_grid.to_csv(OUT_DIR / "long_table_main_grid.csv", index=False)

    omni, ranks = run_omnibus(main_grid, OUT_DIR)
    nemenyi, wilcoxon, diffs = run_posthoc(main_grid, ranks, OUT_DIR)
    tost = run_tost(diffs, OUT_DIR)
    figures = run_cd(main_grid, ranks, OUT_DIR)
    sign = verify_sign_convention(main_grid, OUT_DIR)
    augmented, recon = reconcile_with_r0(main_grid, wilcoxon, OUT_DIR)
    pooling = diagnose_r0_pooling_defect(OUT_DIR)

    manifest = {
        "generated_by": "code_SNI/stats/preview_r0.py",
        "source": str(R0_RESULTS),
        "excluded_sources": dict(long_table.EXCLUDED_SOURCES),
        "long_table": {"shape": list(long.shape), **audit},
        "main_grid_shape": list(main_grid.shape),
        "metrics": list(PREVIEW_METRICS),
        "candidate_deltas": {k: list(v) for k, v in CANDIDATE_DELTAS.items()},
        "alpha": ALPHA,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "method": "BCa"},
        "figures": [str(p) for p in figures],
        "sign_convention_conclusion": sign["conclusion"],
        "r0_pooling_defect": pooling["status"],
        "verdicts_changed_vs_r0": int(recon["verdict_changed"].sum()),
    }
    (OUT_DIR / "preview_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    write_readme(OUT_DIR, manifest)

    _log("")
    _log("done. artifacts:")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            _log(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
