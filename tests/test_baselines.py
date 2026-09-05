"""Regression tests for the de-leaked baselines (P1 T1.4 step 5).

Two properties are asserted, and they are the two that reviewers R1-2 (leakage)
and R1-3 (fit/transform) actually asked about:

**No oracle access.**
    Build two ground-truth frames that agree on every *observed* cell and differ
    wildly on every *masked* cell (absurd sentinel values). A de-leaked imputer
    sees only ``X_missing``, which is identical in both worlds, so its output
    must be bit-identical. Any surviving read of the ground truth would make the
    two outputs differ. The same test is run against ``legacy_oracle=True`` as a
    positive control: it is *expected* to differ, which proves the test has
    power and is not vacuously passing.

**No train -> test information flow.**
    ``fit`` on one frame and ``transform`` on a disjoint frame must not depend on
    rows the imputer was never fitted on: perturbing rows of the test frame that
    are then discarded, or extending the test frame with extra rows, must not
    change the imputed values of the retained rows.

Run with::

    PYTHONPATH=$PWD \\
        python -m pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from baselines import (  # noqa: E402
    FIT_TRANSFORM_ADJUDICATION,
    ORACLE_USAGE_R0,
    DataSchema,
    ObservedStats,
    build_baseline_imputer,
    list_baselines,
)

warnings.filterwarnings("ignore")

CAT_VARS = ["c1", "c2"]
CONT_VARS = ["x1", "x2", "x3"]

#: Methods whose deterministic behavior makes an exact-equality assertion safe.
#: The stochastic deep methods (GAIN/MIWAE/TabCSDI) are covered by the separate
#: statistics-level test below, which is exact regardless of the training RNG.
DETERMINISTIC_METHODS = ["MeanMode", "KNN", "MICE", "MissForest"]

SEPARABLE_METHODS = [m for m, v in FIT_TRANSFORM_ADJUDICATION.items() if v["separable"]]
NON_SEPARABLE_METHODS = [m for m, v in FIT_TRANSFORM_ADJUDICATION.items() if not v["separable"]]

METHOD_KWARGS = {
    "MeanMode": {},
    "KNN": {"k": 3},
    "MICE": {"max_iter": 2},
    "MissForest": {"n_estimators": 20, "max_iter": 2, "n_jobs": 1},
    "HyperImpute": {"timeout": 120},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frames(n: int = 120, seed: int = 0, miss_rate: float = 0.25):
    """Return (X_complete, X_missing, X_complete_sentinel, schema).

    ``X_complete_sentinel`` agrees with ``X_complete`` wherever the mask keeps a
    cell observed, and carries absurd values everywhere the mask hides one.
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(n)
    X = pd.DataFrame({
        "x1": z * 2.0 + 10.0,
        "x2": rng.randn(n) * 5.0 + z,
        "x3": rng.gamma(2.0, 3.0, size=n),
        "c1": pd.Series(rng.randint(0, 3, size=n)).astype("Int64"),
        "c2": pd.Series((z > 0).astype(int)).astype("Int64"),
    })[CONT_VARS + CAT_VARS]

    mask = pd.DataFrame(
        rng.rand(n, X.shape[1]) < miss_rate, columns=X.columns, index=X.index
    )
    # Guarantee every column keeps at least a few observed rows and at least one
    # missing cell, so all code paths are exercised.
    for c in X.columns:
        mask.iloc[:5, mask.columns.get_loc(c)] = False
        mask.iloc[5, mask.columns.get_loc(c)] = True

    X_missing = X.mask(mask)

    X_sentinel = X.copy()
    for c in CONT_VARS:
        X_sentinel.loc[mask[c], c] = -9.99e6
    for c in CAT_VARS:
        X_sentinel[c] = X_sentinel[c].astype("Int64")
        # -777 deliberately sorts BEFORE every observed level. Category sets are
        # sorted (R0 utils.py:41-45, R1 schema._sorted_unique), so an oracle
        # vocabulary that contains it shifts every integer code by one. A
        # sentinel that sorted last would leave the codes intact and the
        # vocabulary leak would look numerically inert -- which is exactly the
        # false reassurance we must not build into the test.
        X_sentinel.loc[mask[c], c] = -777

    schema = DataSchema.from_var_lists(CAT_VARS, CONT_VARS, dataset="_synthetic")
    return X, X_missing, X_sentinel, schema


@pytest.fixture(scope="module")
def frames():
    return _make_frames()


# ---------------------------------------------------------------------------
# 1. No oracle access
# ---------------------------------------------------------------------------

def test_registry_exposes_all_eight_baselines():
    assert list_baselines() == sorted(
        ["MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE",
         "HyperImpute", "TabCSDI"]
    )


def test_impute_signature_has_no_x_complete():
    """The primary interface must not accept a ground-truth frame at all."""
    import inspect

    from baselines.registry import BaseBaseline

    params = list(inspect.signature(BaseBaseline.impute).parameters)
    assert params == ["self", "X_missing", "schema"], params
    assert "X_complete" not in params


@pytest.mark.parametrize("method", DETERMINISTIC_METHODS)
def test_deleaked_output_is_invariant_to_ground_truth(frames, method):
    """Swapping the truth table for a sentinel table must not change anything.

    ``BaseBaseline.run`` is the only entry point that still *accepts* a complete
    frame (the impact study needs it to drive the legacy path). With
    ``legacy_oracle=False`` it must discard the argument entirely, so feeding it
    a table whose every masked cell is -9.99e6 / category 777 must produce
    bit-identical output.
    """
    X_complete, X_missing, X_sentinel, schema = frames
    kwargs = METHOD_KWARGS[method]

    a = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=7, **kwargs)
    out_true = a.run(X_missing, schema=schema, X_complete=X_complete)

    b = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=7, **kwargs)
    out_sentinel = b.run(X_missing, schema=schema, X_complete=X_sentinel)

    pd.testing.assert_frame_equal(out_true, out_sentinel, check_dtype=False)

    # And the observed cells must be preserved verbatim.
    obs = X_missing.notna()
    for c in X_missing.columns:
        left = out_true.loc[obs[c], c]
        right = X_missing.loc[obs[c], c]
        assert (pd.to_numeric(left, errors="coerce").to_numpy()
                == pd.to_numeric(right, errors="coerce").to_numpy()).all(), c


@pytest.mark.parametrize("method", sorted(ORACLE_USAGE_R0.keys() - {"_ALL"}))
def test_every_method_fits_statistics_on_the_incomplete_table(frames, method):
    """Covers all eight methods, including the stochastic deep ones.

    Whatever the training RNG does, the sufficient statistics the imputer was
    allowed to see must be exactly those of ``X_missing`` -- and demonstrably not
    those of the ground truth.
    """
    X_complete, X_missing, X_sentinel, schema = frames

    imp = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=5)
    stats_used = imp._fit_stats(X_missing, schema)

    expected = ObservedStats.from_frame(X_missing, schema)
    assert stats_used.to_dict() == expected.to_dict()
    assert stats_used.source == "observed"

    oracle = ObservedStats.from_frame(X_complete, schema, source="oracle")
    assert stats_used.cont_mean != oracle.cont_mean, (
        "observed and oracle statistics coincide, so this dataset cannot "
        "discriminate a leak"
    )
    assert stats_used.cont_range != oracle.cont_range


#: Methods whose R0 leak fed real numbers (means, modes, ranges) into the
#: imputation and must therefore react to a poisoned ground-truth table.
NUMERIC_LEAK_METHODS = ["MeanMode", "KNN", "MICE"]


def _legacy_pair(method, kwargs, X_complete, X_sentinel, X_missing):
    a = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=7,
                               legacy_oracle=True, **kwargs)
    out_true = a.impute_legacy(X_complete, X_missing)
    b = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=7,
                               legacy_oracle=True, **kwargs)
    out_sent = b.impute_legacy(X_sentinel, X_missing)
    miss = X_missing.isna()
    differs = False
    for c in CONT_VARS:
        if miss[c].any():
            lhs = pd.to_numeric(out_true.loc[miss[c], c], errors="coerce").to_numpy()
            rhs = pd.to_numeric(out_sent.loc[miss[c], c], errors="coerce").to_numpy()
            if not np.allclose(lhs, rhs, equal_nan=True):
                differs = True
    return differs


@pytest.mark.parametrize("method", NUMERIC_LEAK_METHODS)
def test_legacy_path_does_depend_on_ground_truth(frames, method):
    """Positive control: the R0 path must react to the sentinel table.

    Without this, :func:`test_deleaked_output_is_invariant_to_ground_truth`
    could pass vacuously -- e.g. if the sentinel construction stopped being
    discriminative, or if the legacy path were accidentally wired to the
    de-leaked code.
    """
    X_complete, X_missing, X_sentinel, schema = frames
    assert _legacy_pair(method, METHOD_KWARGS[method], X_complete, X_sentinel, X_missing), (
        f"{method}: the legacy oracle path produced identical output for a "
        "ground-truth table poisoned at every masked cell. Either the leak is "
        "already absent for this method or the sentinel is not discriminative."
    )


def test_missforest_vocabulary_leak_is_numerically_inert(frames):
    """MissForest is the one documented case where the R0 leak has no effect.

    Its only oracle exposure was the categorical vocabulary installed by
    ``registry.py:152``. Widening that vocabulary can only reorder integer codes
    monotonically (category sets are sorted), and a decision tree is invariant
    under order-preserving transforms of a feature: same bootstrap indices, same
    candidate splits, same partition. The extra levels also never appear in
    ``y_train``, so no classifier can emit them.

    This is asserted rather than merely claimed, because it is the justification
    for reporting a delta of exactly zero for MissForest in the impact table.
    """
    X_complete, X_missing, X_sentinel, schema = frames
    differs = _legacy_pair("MissForest", METHOD_KWARGS["MissForest"],
                           X_complete, X_sentinel, X_missing)
    assert not differs, (
        "MissForest reacted to the poisoned ground truth; the claim that its "
        "vocabulary-only leak is numerically inert no longer holds and the "
        "impact table must be revisited."
    )


def test_observed_stats_never_see_masked_cells(frames):
    """ObservedStats must be a function of the observed cells only.

    Poison every masked cell of a *filled* copy of the table and re-mask it: the
    statistics must be unchanged, because the poisoned values sit exactly where
    the mask hides them.
    """
    X_complete, X_missing, X_sentinel, schema = frames
    mask = X_missing.isna()

    s1 = ObservedStats.from_frame(X_missing, schema)

    poisoned = X_sentinel.mask(mask)  # observed cells kept, masked cells -> NaN
    s2 = ObservedStats.from_frame(poisoned, schema)
    assert s1.to_dict() == s2.to_dict()

    s_oracle = ObservedStats.from_frame(X_sentinel, schema, source="oracle")
    assert s_oracle.cont_mean != s1.cont_mean, (
        "sentinel ground truth must produce different statistics, otherwise the "
        "oracle/observed contrast is not measurable"
    )


def test_fallback_fillna_uses_observed_statistics(frames):
    from baselines.utils import fallback_fillna, fallback_fillna_oracle

    X_complete, X_missing, X_sentinel, schema = frames
    stats = ObservedStats.from_frame(X_missing, schema)

    partial = X_missing.copy()  # deliberately still full of NaN
    r1 = fallback_fillna(partial, stats, CAT_VARS, CONT_VARS)
    assert r1.isna().sum().sum() == 0
    for c in CONT_VARS:
        filled = pd.to_numeric(r1.loc[X_missing[c].isna(), c]).unique()
        assert len(filled) == 1
        assert np.isclose(filled[0], stats.cont_mean[c])

    r2 = fallback_fillna_oracle(partial, X_sentinel, CAT_VARS, CONT_VARS)
    for c in CONT_VARS:
        v1 = pd.to_numeric(r1.loc[X_missing[c].isna(), c]).iloc[0]
        v2 = pd.to_numeric(r2.loc[X_missing[c].isna(), c]).iloc[0]
        assert not np.isclose(v1, v2), c


# ---------------------------------------------------------------------------
# 2. fit / transform: no train -> test information flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["MeanMode", "KNN", "MICE", "MissForest"])
def test_transform_ignores_rows_outside_the_query(frames, method):
    """Adding extra rows to the test frame must not change the retained rows.

    A transductive imputer pools the whole frame, so its answer for row i moves
    when row j is added. A genuine transform does not.
    """
    X_complete, X_missing, X_sentinel, schema = frames
    kwargs = METHOD_KWARGS[method]

    train = X_missing.iloc[:80].reset_index(drop=True)
    test_a = X_missing.iloc[80:100].reset_index(drop=True)
    test_b = X_missing.iloc[80:].reset_index(drop=True)  # 20 rows + 20 more

    imp = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=11, **kwargs)
    imp.fit(train, schema)
    out_a = imp.transform(test_a)
    out_b = imp.transform(test_b).iloc[: len(test_a)].reset_index(drop=True)

    pd.testing.assert_frame_equal(out_a, out_b, check_dtype=False)


@pytest.mark.parametrize("method", ["MeanMode", "KNN", "MICE", "MissForest"])
def test_transform_state_is_fixed_by_fit(frames, method):
    """Re-fitting on a different training fold must change transform output.

    Positive control for the previous test: it shows the fitted state is really
    what drives ``transform`` (otherwise "transform ignores extra rows" could
    pass for a trivial imputer that ignores everything).
    """
    X_complete, X_missing, X_sentinel, schema = frames
    kwargs = METHOD_KWARGS[method]
    test = X_missing.iloc[100:].reset_index(drop=True)

    imp1 = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=11, **kwargs)
    imp1.fit(X_missing.iloc[:50].reset_index(drop=True), schema)
    out1 = imp1.transform(test)

    imp2 = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=11, **kwargs)
    # Shift the training fold's continuous columns; the fitted statistics and
    # models must move with it.
    train2 = X_missing.iloc[:50].reset_index(drop=True).copy()
    for c in CONT_VARS:
        train2[c] = train2[c] + 25.0
    imp2.fit(train2, schema)
    out2 = imp2.transform(test)

    miss = test.isna()
    moved = any(
        not np.allclose(
            pd.to_numeric(out1.loc[miss[c], c], errors="coerce").to_numpy(),
            pd.to_numeric(out2.loc[miss[c], c], errors="coerce").to_numpy(),
        )
        for c in CONT_VARS if miss[c].any()
    )
    assert moved, f"{method}: transform output did not react to the fitted state"


def test_hyperimpute_package_fit_transform_is_a_facade():
    """Records why HyperImpute is adjudicated NOT separable.

    ``HyperImputePlugin`` advertises the sklearn estimator API but
    ``plugin_hyperimpute.py:124-125`` defines ``_fit`` as ``return self`` and
    ``:127-128`` defines ``_transform`` as ``self.model.fit_transform(X)``.
    ``fit`` therefore stores nothing and ``transform`` reruns the whole AutoML
    search on the frame it is handed.

    Asserted by source inspection (fast and version-explicit) rather than by
    running the search, which takes minutes. If a future package version
    implements real separation, this test fails and the adjudication in
    ``registry.FIT_TRANSFORM_ADJUDICATION`` must be revisited.
    """
    import inspect

    from hyperimpute.plugins.imputers.plugin_hyperimpute import HyperImputePlugin

    fit_src = inspect.getsource(HyperImputePlugin._fit)
    tr_src = inspect.getsource(HyperImputePlugin._transform)
    assert "return self" in fit_src and len(fit_src.strip().splitlines()) <= 3, fit_src
    assert "fit_transform" in tr_src, tr_src


@pytest.mark.parametrize("method", NON_SEPARABLE_METHODS)
def test_non_separable_methods_refuse_fit_transform(frames, method):
    """GAIN / MIWAE / TabCSDI must fail loudly rather than fake a transform."""
    X_complete, X_missing, X_sentinel, schema = frames
    imp = build_baseline_imputer(method, CAT_VARS, CONT_VARS, seed=3)
    assert imp.supports_fit_transform is False
    with pytest.raises(NotImplementedError):
        imp.fit(X_missing, schema)
    with pytest.raises(NotImplementedError):
        imp.transform(X_missing)


@pytest.mark.parametrize("method", SEPARABLE_METHODS)
def test_separable_methods_declare_support(method):
    assert FIT_TRANSFORM_ADJUDICATION[method]["separable"] is True
    assert FIT_TRANSFORM_ADJUDICATION[method]["fitted"]
    assert FIT_TRANSFORM_ADJUDICATION[method]["transformed"]


@pytest.mark.parametrize("method", NON_SEPARABLE_METHODS)
def test_non_separable_methods_document_the_reason(method):
    reason = FIT_TRANSFORM_ADJUDICATION[method]["reason"]
    assert reason and len(reason) > 200, method


@pytest.mark.parametrize(
    "dataset,cat,cont",
    [
        # ComCri and Concrete are unchanged by T2.1 apart from Concrete's
        # Duration reclassification, so they still pin the YAML against R0's
        # manifest. MIMIC and NHANES deliberately no longer match it: MIMIC's
        # table was replaced wholesale (B36/B64) and NHANES was rebuilt with the
        # four missing XPT modules (B61), which is the whole point of T2.1. A
        # test asserting they still match R0 would be asserting the data-layer
        # work did not happen.
        # ViolentCrimesPerPop is the downstream target and is not a feature
        # (Q1-4); every other column, drivers included, reaches the imputer.
        ("ComCri",
         ["IncomeLevel", "UrbanType", "EducationLevel", "CrimeLevel", "RegionCode"],
         ["medIncome", "PctUnemployed", "PctFam2Par", "PctPopUnderPov"]),
    ],
)
def test_schema_from_yaml_matches_r0_manifest(dataset, cat, cont):
    """E1: configs/datasets.yaml must reproduce the R0 manifest variable lists.

    If it did not, the de-leaked runs and the legacy runs would be imputing
    different column sets and the impact table would be meaningless.

    Scoped to the datasets T2.1 left alone -- see the parametrisation note.
    """
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", dataset)
    assert sorted(schema.categorical_vars) == sorted(cat)
    assert sorted(schema.continuous_vars) == sorted(cont)
    # The YAML's numeric metadata must not be wired into the imputer path.
    assert schema.declared_categories == {}, (
        "declared_categories is an external-codebook hook; populating it from "
        "data-derived values would reopen the leak"
    )


def test_oracle_usage_table_covers_every_method():
    for m in list_baselines():
        assert m in ORACLE_USAGE_R0, m
        entry = ORACLE_USAGE_R0[m]
        assert entry["r0_evidence"] and entry["r1_replacement"]
    assert "_ALL" in ORACLE_USAGE_R0  # the shared fallback_fillna leak
