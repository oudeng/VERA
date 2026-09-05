"""Unit tests for :mod:`stats` and :mod:`evaluation` (task T1.7).

Run with::

    PYTHONPATH=$PWD \
        python -m pytest code_SNI/tests/test_stats.py -q

Tests that need the frozen R0 result tree are skipped when it is absent, so the
file remains runnable outside this workspace.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import metrics as ev_metrics
from evaluation import protocol as ev_protocol
from sni import metrics as sni_metrics
from stats import cd_diagram, effect_size, equivalence, intervals, long_table, omnibus, posthoc

#: P7-A closeout: no private absolute path in a published file. The
#: R0 tree is not in this repository (it holds restricted derived
#: tables); point at it with SNI_R0_ROOT, and default to the sibling
#: directory a full checkout would have. A clone that lacks it gets a
#: path it can act on rather than a stranger's home directory.
R0_ROOT = Path(os.environ.get("SNI_R0_ROOT",
                    Path(__file__).resolve().parents[2]
                    / "project_sni_R0"))
R0_RESULTS = R0_ROOT / "results_all"
R0_AVAILABLE = (R0_RESULTS / "agg_sni_v03_main" / "summary_all.csv").exists()
needs_r0 = pytest.mark.skipif(not R0_AVAILABLE, reason="frozen R0 result tree not available")


# =========================================================================== #
# evaluation.metrics -- exclude-columns mode
# =========================================================================== #


def _toy_frames(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 120
    X_complete = pd.DataFrame(
        {
            "a": rng.normal(10.0, 3.0, n),
            "b": rng.normal(0.0, 1.0, n),
            "constant_col": np.full(n, 24.0),           # the B35 pattern
            "huge_scale": rng.normal(1e5, 5e4, n),      # the B47 pattern
            "cat1": rng.integers(0, 3, n),
            "cat2": rng.integers(0, 4, n),
        }
    )
    mask = pd.DataFrame(rng.random((n, X_complete.shape[1])) < 0.3, columns=X_complete.columns)
    X_missing = X_complete.mask(mask)
    X_imputed = X_complete + rng.normal(0.0, 0.4, X_complete.shape) * (
        X_complete.std().to_numpy()[None, :]
    )
    X_imputed[["cat1", "cat2"]] = X_complete[["cat1", "cat2"]].where(
        rng.random((n, 2)) < 0.7, (X_complete[["cat1", "cat2"]] + 1)
    )
    return X_complete, X_missing, X_imputed


CONT = ["a", "b", "constant_col", "huge_scale"]
CAT = ["cat1", "cat2"]


def test_metrics_passthrough_is_identical_to_r0():
    """With no exclusions the wrapper must reproduce sni.metrics exactly."""
    Xc, Xm, Xi = _toy_frames()
    ref = sni_metrics.evaluate_imputation(Xi, Xc, Xm, CAT, CONT)
    got = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT)
    assert set(ref.summary) == set(got.summary)
    for k in ref.summary:
        assert ref.summary[k] == pytest.approx(got.summary[k], rel=0, abs=0)
    pd.testing.assert_frame_equal(ref.per_feature, got.per_feature)


def test_exclusion_leaves_retained_columns_untouched():
    """Dropping a column must not perturb any other column's per-feature value."""
    Xc, Xm, Xi = _toy_frames()
    full = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT)
    dropped = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT, exclude_cols=["constant_col"])

    f = full.per_feature.set_index("feature")
    d = dropped.per_feature.set_index("feature")
    assert "constant_col" not in d.index
    for col in d.index:
        for m in ("NRMSE", "RMSE", "MAE", "R2", "Accuracy", "Macro-F1"):
            if m in f.columns and not pd.isna(f.loc[col, m]):
                assert f.loc[col, m] == pytest.approx(d.loc[col, m], rel=0, abs=0)


def test_exclusion_changes_summary_only_via_the_feature_average():
    Xc, Xm, Xi = _toy_frames()
    full = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT)
    dropped = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT, exclude_cols=["constant_col"])
    pf = full.per_feature.set_index("feature")
    kept = [c for c in CONT if c != "constant_col"]
    expected = float(np.nanmean(pf.loc[kept, "NRMSE"].to_numpy(dtype=float)))
    assert dropped.summary["cont_NRMSE"] == pytest.approx(expected)
    assert dropped.summary["n_cont_features"] == len(kept)


def test_nrmse_uses_the_complete_column_range():
    """R0 definition: the denominator is the range of the COMPLETE column."""
    Xc, Xm, Xi = _toy_frames()
    res = ev_metrics.evaluate(Xi, Xc, Xm, [], ["a"])
    pf = res.per_feature.set_index("feature")
    mask = Xm["a"].isna()
    true_vals = Xc.loc[mask, "a"].to_numpy(dtype=float)
    pred_vals = Xi.loc[mask, "a"].to_numpy(dtype=float)
    rmse = math.sqrt(float(np.mean((true_vals - pred_vals) ** 2)))
    full_range = float(Xc["a"].max() - Xc["a"].min())
    subset_range = float(true_vals.max() - true_vals.min())
    assert pf.loc["a", "NRMSE"] == pytest.approx(rmse / full_range)
    assert pf.loc["a", "NRMSE"] != pytest.approx(rmse / subset_range)


def test_evaluation_is_on_masked_cells_only():
    Xc, Xm, Xi = _toy_frames()
    res = ev_metrics.evaluate(Xi, Xc, Xm, CAT, CONT)
    pf = res.per_feature.set_index("feature")
    for col in CONT + CAT:
        assert int(pf.loc[col, "n_eval"]) == int(Xm[col].isna().sum())


def test_unknown_exclusion_column_raises():
    with pytest.raises(ValueError, match="absent from the declared roles"):
        ev_metrics.resolve_variable_lists(CAT, CONT, ["not_a_column"])


def test_evaluate_many_shape():
    Xc, Xm, Xi = _toy_frames()
    specs = [
        ev_metrics.EvaluationSpec(CAT, CONT, name="full"),
        ev_metrics.EvaluationSpec(CAT, CONT, exclude_cols=["constant_col"], name="drop_B35"),
        ev_metrics.EvaluationSpec(CAT, CONT, exclude_cols=["huge_scale"], name="drop_B47"),
    ]
    out = ev_metrics.evaluate_many(Xi, Xc, Xm, specs)
    assert set(out["spec"]) == {"full", "drop_B35", "drop_B47"}
    assert {"spec", "excluded", "metric", "value"} <= set(out.columns)


# =========================================================================== #
# evaluation.protocol
# =========================================================================== #


class _InductiveMeanImputer:
    """Column means learned on fit, applied on transform."""

    def __init__(self):
        self.means_ = None

    def fit(self, X):
        self.means_ = X.mean(numeric_only=True)
        return self

    def transform(self, X):
        return X.fillna(self.means_)


class _TransductiveMeanImputer:
    """Fills with the means of whatever matrix it is handed -- no transform."""

    def impute(self, X_missing):
        return X_missing.fillna(X_missing.mean(numeric_only=True))


def _missing_frame(seed: int = 1, n: int = 60):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("pqrs"))
    return X.mask(rng.random(X.shape) < 0.25)


def test_protocol_detection():
    assert ev_protocol.detect_protocol(_InductiveMeanImputer()) == ev_protocol.PROTOCOL_FIT_TRANSFORM
    assert ev_protocol.detect_protocol(_TransductiveMeanImputer()) == ev_protocol.PROTOCOL_FOLD_INDEPENDENT


def test_fit_transform_uses_only_training_statistics():
    X = _missing_frame()
    folds = ev_protocol.make_holdout_folds(len(X), test_size=0.3, seed=0)
    rep = ev_protocol.run_protocol("mean", _InductiveMeanImputer, X, folds)
    f = rep.folds[0]
    train_means = X.iloc[f.train_idx].mean()
    filled = f.test_imputed["p"][X.iloc[f.test_idx]["p"].isna().to_numpy()]
    assert len(filled) > 0
    assert np.allclose(filled.to_numpy(dtype=float), float(train_means["p"]))


def test_fold_independent_fallback_does_not_use_the_train_fold():
    X = _missing_frame()
    folds = ev_protocol.make_holdout_folds(len(X), test_size=0.3, seed=0)
    rep = ev_protocol.run_protocol("mean_t", _TransductiveMeanImputer, X, folds)
    assert rep.protocol == ev_protocol.PROTOCOL_FOLD_INDEPENDENT
    f = rep.folds[0]
    test_means = X.iloc[f.test_idx].mean()
    filled = f.test_imputed["p"][X.iloc[f.test_idx]["p"].isna().to_numpy()]
    assert np.allclose(filled.to_numpy(dtype=float), float(test_means["p"]))


@pytest.mark.parametrize("factory", [_InductiveMeanImputer, _TransductiveMeanImputer])
def test_verify_fold_independence_passes_for_both_protocols(factory):
    X = _missing_frame()
    folds = ev_protocol.make_holdout_folds(len(X), test_size=0.3, seed=0)
    out = ev_protocol.verify_fold_independence(factory, X, folds[0])
    assert out["deterministic_reference"] is True
    if factory is _TransductiveMeanImputer:
        assert out["independent"] is True, "fold-independent path leaked train information"
    else:
        # Inductive imputers are *supposed* to depend on the train fold; the
        # test-fold OUTPUT changes, which is the expected behavior. What must
        # never happen is the reverse (train depending on test), covered above.
        assert out["n_cell_changes"] > 0


def test_r0_leakage_reference_is_documented():
    ev = ev_protocol.R0_LEAKAGE_EVIDENCE
    assert ev["path"].endswith("exp2_downstream_task_validation.py")
    assert ev["split_after_impute"] == "517-520"


def test_fold_spec_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        ev_protocol.FoldSpec(0, np.array([0, 1, 2]), np.array([2, 3]))


def test_fold_blocks_are_positionally_anonymous():
    """The imputer must never see the row's position in the original table."""
    seen = {}

    class _IndexSpy:
        def fit(self, X):
            seen["fit"] = list(X.index)
            return self

        def transform(self, X):
            seen.setdefault("transform", []).append(list(X.index))
            return X.fillna(0.0)

    X = _missing_frame()
    folds = ev_protocol.make_holdout_folds(len(X), test_size=0.3, seed=0)
    rep = ev_protocol.run_protocol("spy", _IndexSpy, X, folds)
    assert seen["fit"] == list(range(len(folds[0].train_idx)))
    for idx in seen["transform"]:
        assert idx == list(range(len(idx)))
    # ...but the caller gets the original index back.
    assert list(rep.folds[0].test_imputed.index) == list(X.index[folds[0].test_idx])


# --------------------------------------------------------------------------- #
# Interop with the R1 baseline registry (owned by task T1.4; skipped if absent)
# --------------------------------------------------------------------------- #

FAST_INDUCTIVE = ["MeanMode", "KNN", "MICE", "MissForest"]


@pytest.mark.parametrize("method", FAST_INDUCTIVE)
def test_r1_baselines_run_under_the_within_fold_protocol(method):
    registry = pytest.importorskip("baselines.registry")
    X = _missing_frame(seed=2, n=80)
    X["cat"] = np.where(np.isnan(X["p"]), np.nan, np.floor(X["p"].abs() * 2).clip(0, 2))
    folds = ev_protocol.make_holdout_folds(len(X), test_size=0.3, seed=0)

    def factory():
        return registry.build_baseline_imputer(
            method, categorical_vars=["cat"], continuous_vars=list("pqrs"), seed=1
        )

    rep = ev_protocol.run_protocol(method, factory, X, folds)
    assert rep.protocol == ev_protocol.PROTOCOL_FIT_TRANSFORM
    f = rep.folds[0]
    assert f.train_imputed.isna().sum().sum() == 0
    assert f.test_imputed.isna().sum().sum() == 0
    assert list(f.test_imputed.index) == list(X.index[folds[0].test_idx])


def test_transductive_baselines_are_classified_as_fold_independent():
    registry = pytest.importorskip("baselines.registry")
    for method in ("GAIN", "MIWAE", "TabCSDI"):
        imp = registry.build_baseline_imputer(
            method, categorical_vars=["c"], continuous_vars=["a", "b"], seed=1
        )
        assert ev_protocol.detect_protocol(imp) == ev_protocol.PROTOCOL_FOLD_INDEPENDENT


# =========================================================================== #
# stats.long_table
# =========================================================================== #


def test_deep_exclusion_is_asserted():
    with pytest.raises(ValueError, match="agg_baselines_deep"):
        long_table.assert_deep_excluded(["agg_sni_v03_main", "agg_baselines_deep"])
    with pytest.raises(ValueError, match="agg_baselines_deep"):
        long_table.build_long_table("/nonexistent", sources=["agg_baselines_deep"])
    # Path-shaped inputs are caught too.
    with pytest.raises(ValueError):
        long_table.assert_deep_excluded([Path("results_all/agg_baselines_deep")])


def test_deep_is_not_in_the_default_source_list():
    assert "agg_baselines_deep" not in long_table.AGG_SOURCES
    assert "agg_baselines_deep" in long_table.EXCLUDED_SOURCES


def test_rate_normalisation_handles_both_schemas():
    assert long_table._normalise_rate("30per", np.nan) == (0.30, "30per")
    assert long_table._normalise_rate(np.nan, 0.5) == (0.5, "50per")
    assert long_table._normalise_rate("10per", 0.1) == (0.1, "10per")


@needs_r0
def test_build_long_table_shape_and_coverage():
    long = long_table.build_long_table(R0_RESULTS)
    audit = long_table.audit_long_table(long)
    assert audit["duplicate_key_rows"] == 0
    assert audit["seeds"] == [1, 2, 3, 5, 8]
    assert audit["seeds_per_cell"] == [5]
    assert audit["cells_with_wrong_seed_count"] == 0
    assert "agg_baselines_deep" not in audit["sources"]
    # 1,430 successful runs are recorded in the seven retained aggregates.
    assert audit["n_runs"] == 1430
    assert audit["n_cells"] == 286


@needs_r0
def test_lambda_ablation_is_tagged_and_never_pooled_with_main_sni():
    long = long_table.build_long_table(R0_RESULTS)
    lam = long[long["experiment_family"] == "lambda_ablation"]
    assert set(lam["method"]) == {"lam0.1", "lam0.5", "lam1.0", "lam2.0", "lam5.0"}
    assert set(lam["algo"]) == {"SNI"}
    main = long_table.main_grid_view(long)
    assert not set(main["method"]) & set(lam["method"])


@needs_r0
def test_main_grid_view_is_the_published_grid():
    long = long_table.build_long_table(R0_RESULTS)
    main = long_table.main_grid_view(long)
    assert main["dataset"].nunique() == 6
    assert set(main["mechanism"]) == {"MCAR", "MAR"}
    assert main["method"].nunique() == 9
    mat = long_table.to_setting_matrix(main, "NRMSE")
    assert mat.shape == (12, 9)


@needs_r0
def test_long_table_reproduces_a_known_r0_cell():
    """Spot-check against the raw wide table."""
    long = long_table.build_long_table(R0_RESULTS)
    wide = pd.read_csv(R0_RESULTS / "agg_sni_v03_main" / "summary_all.csv")
    row = wide[wide["exp_id"] == "V03_MAIN_MIMIC_MAR_30per_SNI_s3"].iloc[0]
    got = long[(long["exp_id"] == "V03_MAIN_MIMIC_MAR_30per_SNI_s3") & (long["metric"] == "NRMSE")]
    assert len(got) == 1
    assert float(got["value"].iloc[0]) == pytest.approx(float(row["cont_NRMSE"]))


# =========================================================================== #
# stats.omnibus
# =========================================================================== #


def _rank_fixture():
    rng = np.random.default_rng(7)
    n, k = 12, 5
    base = rng.normal(0, 1, (n, 1))
    shifts = np.array([0.0, 0.3, 0.6, 0.9, 1.2])[None, :]
    vals = base + shifts + rng.normal(0, 0.25, (n, k))
    return pd.DataFrame(vals, columns=[f"m{i}" for i in range(k)])


def test_friedman_matches_scipy_for_tie_free_data():
    from scipy.stats import friedmanchisquare

    mat = _rank_fixture()
    chi2, df, p, _ = omnibus.friedman_test(mat, higher_is_better=False)
    ref = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    assert chi2 == pytest.approx(float(ref.statistic), rel=1e-9)
    assert p == pytest.approx(float(ref.pvalue), rel=1e-9)
    assert df == mat.shape[1] - 1


def test_rank_direction():
    mat = pd.DataFrame({"good": [1.0, 1.0], "bad": [2.0, 2.0]})
    lower = omnibus.average_ranks(mat, higher_is_better=False)
    assert lower["good"] < lower["bad"]
    higher = omnibus.average_ranks(mat, higher_is_better=True)
    assert higher["bad"] < higher["good"]


def test_iman_davenport_shape_and_power():
    mat = _rank_fixture()
    res = omnibus.friedman_from_matrix(mat, metric="toy", higher_is_better=False)
    assert res.F_df1 == res.k_methods - 1
    assert res.F_df2 == (res.k_methods - 1) * (res.n_blocks - 1)
    # For a clearly separated field the F correction is the more powerful test.
    assert res.F_p <= res.chi2_p
    # Closed form check.
    F, _, _, _ = omnibus.iman_davenport(res.chi2, res.n_blocks, res.k_methods)
    n, k = res.n_blocks, res.k_methods
    assert F == pytest.approx((n - 1) * res.chi2 / (n * (k - 1) - res.chi2))


def test_friedman_rejects_incomplete_blocks():
    mat = _rank_fixture()
    mat.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete block design"):
        omnibus.friedman_test(mat, higher_is_better=False)


# =========================================================================== #
# stats.posthoc
# =========================================================================== #


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    for pvals in (
        [0.001, 0.008, 0.039, 0.041, 0.9],
        [0.00048828125, 0.00048828125, 0.20361328125, 0.67724609375],
        [0.5, 0.5, 0.5],
    ):
        mine = posthoc.holm_bonferroni(pvals)
        ref = multipletests(pvals, method="holm")[1]
        assert np.allclose(mine, ref), f"{mine} != {list(ref)}"


def test_holm_propagates_nan():
    out = posthoc.holm_bonferroni([0.01, float("nan"), 0.02])
    assert math.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


@pytest.mark.parametrize("k,q_expected", [(2, 1.960), (3, 2.343), (5, 2.728), (9, 3.102), (10, 3.164)])
def test_nemenyi_q_matches_demsar_table5(k, q_expected):
    """Demsar (2006) Table 5 lists q_0.05 already divided by sqrt(2)."""
    assert cd_q_alpha(k, 0.05) == pytest.approx(q_expected, abs=0.002)


def test_nemenyi_cd_closed_form():
    k, n, alpha = 5, 12, 0.05
    q = cd_q_alpha(k, alpha)
    cd = posthoc.nemenyi_critical_difference(k, n, alpha)
    assert cd == pytest.approx(q * math.sqrt(k * (k + 1) / (6 * n)), rel=1e-12)


def cd_q_alpha(k: int, alpha: float) -> float:
    from scipy import stats as _st

    return float(_st.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2.0)


def test_bonferroni_dunn_is_tighter_than_nemenyi():
    assert posthoc.bonferroni_dunn_critical_difference(9, 12) < posthoc.nemenyi_critical_difference(9, 12)


def test_wilcoxon_holm_sign_convention_lower_is_better():
    """For a lower-is-better metric, positive mean_diff must mean reference better."""
    mat = pd.DataFrame(
        {"SNI": np.linspace(0.10, 0.12, 12), "Worse": np.linspace(0.20, 0.22, 12)}
    )
    out = posthoc.wilcoxon_holm(mat, "SNI", metric="NRMSE")
    assert len(out) == 1
    assert out["mean_diff"].iloc[0] > 0
    assert out["direction"].iloc[0] == "reference_better"


def test_wilcoxon_holm_sign_convention_higher_is_better():
    mat = pd.DataFrame({"SNI": np.linspace(0.8, 0.9, 12), "Worse": np.linspace(0.5, 0.6, 12)})
    out = posthoc.wilcoxon_holm(mat, "SNI", metric="R2")
    assert out["mean_diff"].iloc[0] > 0
    assert out["direction"].iloc[0] == "reference_better"


def test_wilcoxon_safe_edge_cases():
    assert all(math.isnan(v) for v in posthoc.wilcoxon_safe(np.array([1.0, 2.0])))
    assert posthoc.wilcoxon_safe(np.zeros(10)) == (0.0, 1.0)


R0_S3_NAME_MAP = {"NRMSE": "NRMSE", "R2": "R2", "Spearman_rho": "Spearman", "Macro_F1": "Macro-F1"}


@needs_r0
def test_wilcoxon_holm_agrees_with_r0_table_s3_on_inference():
    """Same W, same p, same verdicts as the published Table S3.

    Two documented divergences: ``mean_diff`` differs in the fourth decimal
    (see ``test_r0_table_s3_pooling_defect_is_reproduced_exactly``), and for the
    categorical metrics the effective pair count is 10, not the 12 the CSV
    records, because Concrete has no categorical columns.
    """
    long = long_table.build_long_table(R0_RESULTS)
    main = long_table.main_grid_view(long)
    r0 = pd.read_csv(R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv")
    for r0_metric, our_metric in R0_S3_NAME_MAP.items():
        mat = long_table.to_setting_matrix(main, our_metric)
        got = posthoc.wilcoxon_holm(mat, "SNI", metric=our_metric).set_index("other")
        ref = r0[r0["metric"] == r0_metric].copy()
        ref["other"] = ref["comparison"].str.replace("SNI vs ", "", regex=False)
        ref = ref.set_index("other")
        expected_n = effect_size.R0_EFFECTIVE_N[r0_metric]
        for other in ref.index:
            assert int(got.loc[other, "n_settings"]) == expected_n
            assert int(ref.loc[other, "n_settings"]) == 12  # R0 always writes 12
            # Every published verdict survives the clean recomputation.
            assert bool(got.loc[other, "significant"]) == bool(ref.loc[other, "significant"])
            # sign agreement: both tables use "positive favors SNI"
            assert np.sign(float(got.loc[other, "mean_diff"])) == np.sign(
                float(ref.loc[other, "mean_diff"])
            )
            assert float(got.loc[other, "mean_diff"]) == pytest.approx(
                float(ref.loc[other, "mean_diff"]), abs=1e-3
            )
            if r0_metric != "Macro_F1":
                # Continuous metrics: identical W and identical adjusted p.
                assert float(got.loc[other, "W_statistic"]) == pytest.approx(
                    float(ref.loc[other, "W_statistic"])
                )
                assert float(got.loc[other, "p_adjusted"]) == pytest.approx(
                    float(ref.loc[other, "p_adjusted"]), rel=1e-9
                )


@needs_r0
def test_r0_n_settings_is_wrong_for_categorical_metrics():
    """Concrete has zero categorical columns, so Macro-F1 has 10 pairs, not 12."""
    long = long_table.build_long_table(R0_RESULTS)
    main = long_table.main_grid_view(long)
    assert long_table.to_setting_matrix(main, "NRMSE").shape[0] == 12
    assert long_table.to_setting_matrix(main, "Macro-F1").shape[0] == 10
    assert "Concrete" not in {i[0] for i in long_table.to_setting_matrix(main, "Macro-F1").index}
    r0 = pd.read_csv(R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv")
    assert set(r0["n_settings"]) == {12}


@needs_r0
def test_r0_table_s3_pooling_defect_is_reproduced_exactly():
    """Documents *why* R0's mean_diff values are off, bit for bit.

    ``ext2/scripts/exp5_significance_tests.py`` builds its SNI arm from
    ``_load_sni_results``, which reads ``agg_sni_v03_main`` **and**
    ``sni_v03_main/*/metrics_summary.csv`` (the same 60 runs, counted twice)
    **and** ``agg_sni_v03_ablation_lambda`` -- whose ``variant`` column says
    ``SNI``, so its 50 fixed-lambda runs are pooled into the SNI arm for the
    MIMIC/MAR and NHANES/MAR settings.  Emulating that pooling (main x2 +
    ablation x1) reproduces the published ``mean_diff`` exactly, which proves
    the diagnosis.
    """
    main_sni = pd.read_csv(R0_RESULTS / "agg_sni_v03_main" / "summary_all.csv")
    abla = pd.read_csv(R0_RESULTS / "agg_sni_v03_ablation_lambda" / "summary_all.csv")
    bmain = pd.read_csv(R0_RESULTS / "agg_baselines_main" / "summary_all.csv")
    bnew = pd.read_csv(R0_RESULTS / "agg_baselines_new" / "summary_all.csv")
    r0 = pd.read_csv(R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv")

    contaminated = pd.concat([main_sni, main_sni, abla], ignore_index=True)

    def setting_mean(df, col):
        return df.groupby(["dataset", "mechanism"])[col].mean()

    col_map = {"NRMSE": "cont_NRMSE", "R2": "cont_R2",
               "Spearman_rho": "cont_Spearman", "Macro_F1": "cat_Macro-F1"}
    lower_better = {"NRMSE"}

    for r0_metric, col in col_map.items():
        sni = setting_mean(contaminated, col)
        for _, row in r0[r0["metric"] == r0_metric].iterrows():
            other = row["comparison"].replace("SNI vs ", "")
            src = bmain if other in set(bmain["algo"]) else bnew
            base = setting_mean(src[src["algo"] == other], col)
            m = pd.concat([sni.rename("s"), base.rename("b")], axis=1).dropna()
            diff = m["s"] - m["b"]
            if r0_metric in lower_better:
                diff = -diff
            assert float(diff.mean()) == pytest.approx(float(row["mean_diff"]), abs=1e-12)


# =========================================================================== #
# stats.effect_size
# =========================================================================== #


def test_rank_biserial_extremes():
    assert effect_size.rank_biserial_from_w(0.0, 12, sign=1.0) == pytest.approx(1.0)
    assert effect_size.rank_biserial_from_w(0.0, 12, sign=-1.0) == pytest.approx(-1.0)
    # Perfectly balanced: W = T/2 = 39 for n = 12.
    assert effect_size.rank_biserial_from_w(39.0, 12, sign=1.0) == pytest.approx(0.0)


def test_rank_biserial_from_w_matches_direct_computation_in_magnitude():
    """|r_rb| from the stored W always equals the directly computed value.

    The *sign* has to be supplied separately, and ``mean_diff`` -- the only sign
    carrier stored in R0's CSV -- can disagree with the rank-sum direction when
    the difference vector is skewed.  Magnitude agreement is exact; sign
    agreement holds whenever the mean and the signed-rank sum point the same
    way, which is checked in the next test on the real R0 data.
    """
    rng = np.random.default_rng(11)
    n_sign_mismatch = 0
    for _ in range(200):
        d = rng.normal(0.4, 1.0, 12)
        if np.allclose(d, 0):
            continue
        w, _p = posthoc.wilcoxon_safe(d)
        from_w = effect_size.rank_biserial_from_w(w, int((d != 0).sum()), sign=float(np.mean(d)))
        direct = effect_size.rank_biserial_from_diffs(d)
        assert abs(from_w) == pytest.approx(abs(direct), abs=1e-9)
        if np.sign(from_w) != np.sign(direct):
            n_sign_mismatch += 1
    # Sign disagreement is rare and only occurs for near-zero effects.
    assert n_sign_mismatch < 20


@needs_r0
def test_rank_biserial_closed_form_exact_on_the_real_grid():
    """The W-and-n closed form reproduces the full recomputation on all 32 cells.

    This is the claim that makes the effect sizes a *zero-recomputation* upgrade:
    given the ``W`` and the pair count a Wilcoxon run already produces, no
    difference vector is needed.  Sign has to come from somewhere; four cells --
    all with p > 0.5, i.e. effects indistinguishable from zero -- have a
    ``mean_diff`` whose sign disagrees with the median and hence with the
    signed-rank sum.  They are asserted explicitly rather than tolerated.
    """
    long = long_table.build_long_table(R0_RESULTS)
    main = long_table.main_grid_view(long)
    sign_mismatch = []
    for _r0_metric, our_metric in R0_S3_NAME_MAP.items():
        mat = long_table.to_setting_matrix(main, our_metric)
        got = posthoc.wilcoxon_holm(mat, "SNI", metric=our_metric)
        for _, row in got.iterrows():
            n_pairs = int(row["n_settings"]) - int(row["n_zero"])
            from_w = effect_size.rank_biserial_from_w(
                row["W_statistic"], n_pairs, sign=row["mean_diff"]
            )
            direct = effect_size.rank_biserial_from_diffs(row["_diff"])
            assert abs(from_w) == pytest.approx(abs(direct), abs=1e-9), (our_metric, row["other"])
            if np.sign(from_w) != np.sign(direct):
                sign_mismatch.append((our_metric, row["other"]))
                assert float(row["p_value"]) > 0.5
    assert sorted(sign_mismatch) == sorted(
        [
            ("NRMSE", "HyperImpute"),
            ("Spearman", "HyperImpute"),
            ("Spearman", "TabCSDI"),
            ("Macro-F1", "HyperImpute"),
        ]
    )


@needs_r0
def test_augment_r0_table_flags_the_pair_count_correction():
    r0 = effect_size.augment_r0_wilcoxon_table(
        R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv"
    )
    assert len(r0) == 32
    # The eight Macro_F1 rows are the ones whose stored n is wrong.
    flagged = r0[r0["n_effective_differs_from_stored"]]
    assert len(flagged) == 8
    assert set(flagged["metric"]) == {"Macro_F1"}
    assert set(flagged["n_effective"]) == {10}
    row = r0[(r0["metric"] == "NRMSE") & (r0["comparison"] == "SNI vs MissForest")].iloc[0]
    assert float(row["rank_biserial"]) == pytest.approx(-1.0)
    assert row["rank_biserial_magnitude"] == "large"


@needs_r0
def test_sign_override_is_honoured():
    fixed = effect_size.augment_r0_wilcoxon_table(
        R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv",
        sign_override={("NRMSE", "SNI vs HyperImpute"): +1.0},
    )
    row = fixed[(fixed["metric"] == "NRMSE") & (fixed["comparison"] == "SNI vs HyperImpute")].iloc[0]
    assert row["sign_source"] == "override"
    assert float(row["rank_biserial"]) > 0


def test_cliffs_delta_bounds():
    assert effect_size.cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert effect_size.cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    assert abs(effect_size.cliffs_delta([1, 2, 3], [1, 2, 3])) == pytest.approx(0.0)
    assert effect_size.cliffs_delta_paired([1, 1, -1, -1]) == pytest.approx(0.0)
    assert effect_size.cliffs_delta_paired([1, 1, 1, -1]) == pytest.approx(0.5)


def test_hedges_correction_shrinks_dz():
    d = np.array([0.5] * 6 + [0.1] * 6)
    dz, gz = effect_size.standardized_paired_mean_difference(d)
    assert abs(gz) < abs(dz)
    assert gz / dz == pytest.approx(1 - 3 / (4 * 11 - 1))


def test_magnitude_labels():
    assert effect_size.interpret_magnitude(0.05) == "negligible"
    assert effect_size.interpret_magnitude(0.2) == "small"
    assert effect_size.interpret_magnitude(0.4) == "medium"
    assert effect_size.interpret_magnitude(0.9) == "large"


@needs_r0
def test_augment_r0_wilcoxon_table_zero_recomputation():
    path = R0_RESULTS / "ext2" / "significance" / "wilcoxon_across_settings.csv"
    out = effect_size.augment_r0_wilcoxon_table(path)
    assert len(out) == 32
    assert out["rank_biserial"].notna().all()
    # W = 0 with a negative mean_diff (SNI vs MissForest, NRMSE) => r_rb = -1.
    row = out[(out["metric"] == "NRMSE") & (out["comparison"] == "SNI vs MissForest")].iloc[0]
    assert float(row["rank_biserial"]) == pytest.approx(-1.0)
    assert row["rank_biserial_magnitude"] == "large"


# =========================================================================== #
# stats.intervals
# =========================================================================== #


def test_bootstrap_ci_covers_the_point_estimate():
    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 1.0, 40)
    ci = intervals.bootstrap_ci(x, np.mean, method="bca", n_resamples=2000)
    assert ci.method_used == "bca"
    assert ci.lower < ci.statistic < ci.upper


def test_bootstrap_ci_degenerate_constant():
    ci = intervals.bootstrap_ci(np.full(10, 2.0))
    assert ci.method_used == "degenerate_constant"
    assert ci.lower == ci.upper == 2.0


def test_paired_difference_ci_sign_convention():
    ref = np.linspace(0.10, 0.12, 12)
    other = np.linspace(0.20, 0.22, 12)
    ci = intervals.paired_difference_ci(ref, other, higher_is_better=False, n_resamples=1000)
    assert ci.statistic > 0  # reference better on a lower-is-better metric
    ci2 = intervals.paired_difference_ci(ref, other, higher_is_better=True, n_resamples=1000)
    assert ci2.statistic < 0


def test_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(5)
    small = intervals.bootstrap_ci(rng.normal(0, 1, 12), n_resamples=2000)
    large = intervals.bootstrap_ci(rng.normal(0, 1, 400), n_resamples=2000)
    assert (large.upper - large.lower) < (small.upper - small.lower)


# =========================================================================== #
# stats.equivalence
# =========================================================================== #


def test_tost_delta_has_no_default():
    with pytest.raises(equivalence.MarginNotSpecified):
        equivalence.tost_paired(np.random.default_rng(0).normal(0, 1, 12))
    with pytest.raises(equivalence.MarginNotSpecified):
        equivalence.tost_wilcoxon_paired(np.random.default_rng(0).normal(0, 1, 12))
    with pytest.raises(equivalence.MarginNotSpecified):
        equivalence.bayesian_correlated_ttest(np.random.default_rng(0).normal(0, 1, 12))


def test_tost_rejects_nonpositive_delta():
    d = np.random.default_rng(0).normal(0, 1, 12)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            equivalence.tost_paired(d, bad)


def test_tost_declares_equivalence_for_a_tight_sample():
    rng = np.random.default_rng(4)
    d = rng.normal(0.0, 0.001, 30)
    assert equivalence.tost_paired(d, 0.05).equivalent is True
    assert equivalence.tost_paired(d, 0.0001).equivalent is False


def test_tost_is_monotone_in_delta():
    rng = np.random.default_rng(9)
    d = rng.normal(0.004, 0.01, 12)
    tbl = equivalence.tost_sensitivity(d, [0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    flags = tbl["equivalent"].tolist()
    # once equivalence is attained, larger margins keep it
    assert flags == sorted(flags)
    assert (tbl["p_tost"].diff().dropna() <= 1e-12).all()


def test_nadeau_bengio_factor():
    assert equivalence.nadeau_bengio_variance_factor(10, 90, 10) == pytest.approx(0.1 + 10 / 90)
    with pytest.raises(ValueError):
        equivalence.nadeau_bengio_variance_factor(10, 0, 10)


def test_bayesian_correlated_ttest_skeleton():
    rng = np.random.default_rng(2)
    d = rng.normal(0.0, 0.01, 12)
    res = equivalence.bayesian_correlated_ttest(d, rope=(-0.05, 0.05), rho=0.1)
    assert res.status == "SKELETON"
    assert res.p_left + res.p_rope + res.p_right == pytest.approx(1.0)
    assert res.decision == "rope"
    with pytest.raises(ValueError, match="rho"):
        equivalence.bayesian_correlated_ttest(d, rope=(-0.05, 0.05))


def test_bayesian_correction_widens_the_posterior():
    rng = np.random.default_rng(6)
    d = rng.normal(0.02, 0.01, 12)
    tight = equivalence.bayesian_correlated_ttest(d, rope=(-0.01, 0.01), rho=0.0)
    loose = equivalence.bayesian_correlated_ttest(d, rope=(-0.01, 0.01), rho=0.5)
    assert loose.variance_factor > tight.variance_factor
    assert loose.p_rope > tight.p_rope  # more uncertainty => more mass near zero


# =========================================================================== #
# stats.cd_diagram
# =========================================================================== #


def test_cliques_are_maximal_and_exclude_singletons():
    ranks = pd.Series({"a": 1.0, "b": 1.2, "c": 3.0, "d": 3.1, "e": 8.0})
    cl = cd_diagram.find_cliques(ranks, cd=0.5)
    assert cl == [(0, 1), (2, 3)]
    assert cd_diagram.find_cliques(ranks, cd=0.01) == []
    assert cd_diagram.find_cliques(ranks, cd=100.0) == [(0, 4)]


def test_cd_diagram_writes_a_file(tmp_path):
    ranks = pd.Series({"SNI": 4.27, "MissForest": 2.1, "MeanMode": 7.9, "KNN": 4.9})
    out = tmp_path / "cd.pdf"
    fig = cd_diagram.plot_cd_diagram(ranks, cd=1.8, out_path=out, highlight=["SNI"])
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "cd.png").exists()
    import matplotlib.pyplot as plt

    plt.close(fig)
