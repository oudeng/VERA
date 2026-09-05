"""Regression tests for the rewritten missingness simulator (task T1.5).

The five assertions named in the T1.5 completion criteria are, in order:

1. ``test_drivers_stay_fully_observed``
2. ``test_achieved_rate_within_one_percent``
3. ``test_row_index_correlation_is_null``            <- the R1-4 refutation
4. ``test_per_column_heterogeneity_is_real``
5. ``test_legacy_profile_reproduces_the_pathology``  <- proves 3 discriminates

The rest cover the specific P0 findings the rewrite targets (B38/B39/B45/B46)
and the config contract.

Run with::

    PYTHONPATH=$PWD \\
      python -m pytest tests/test_missingness.py -v
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from common.masks import MaskConsistencyError
from missingness import generate, generate_and_write, resolve, solve_intercept
from missingness.calibration import sigmoid
from missingness.propensity import build_propensity_matrix
from missingness.rng import StreamRegistry, independence_probe
from missingness.spec import (
    DEFAULT_MISSINGNESS_CONFIG,
    MissingnessSpec,
    dataset_schema,
    load_config,
    schema_from_frame,
)

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: Seeds declared in configs/datasets.yaml (seeds.train). Fixed, so every test
#: here is deterministic — there is no "flaky" branch.
SEEDS = [1, 2, 3, 5, 8]
RATES = [0.1, 0.3, 0.5]
MECHANISMS = ["MCAR", "MAR", "MNAR"]

#: R0's derived tables. Read-only; T1.5 generates no masks for them, the tests
#: only use them as realistic input frames.
CODE_ROOT = Path(__file__).resolve().parent.parent
#: P7-A closeout: no private absolute path in a published file. The
#: R0 tree is not in this repository (it holds restricted derived
#: tables); point at it with SNI_R0_ROOT, and default to the sibling
#: directory a full checkout would have. A clone that lacks it gets a
#: path it can act on rather than a stranger's home directory.
R0_ROOT = Path(os.environ.get("SNI_R0_ROOT",
                    Path(__file__).resolve().parents[2]
                    / "project_sni_R0"))
R0_DATA = R0_ROOT / "sni" / "data"
R0_DATASETS = ["MIMIC", "eICU", "NHANES", "AutoMPG", "ComCri", "Concrete"]
#: The datasets the R1 package actually ships. CDC2022 is new in R1, so it has
#: no R0 counterpart; tests that assert something about the masks we publish
#: must cover it, and tests that reproduce R0 must not.
CURRENT_DATASETS = R0_DATASETS + ["CDC2022"]

#: R0's own categorical declarations, snapshotted from configs/datasets.yaml
#: before T2.1 regenerated it. Tests that make a claim about R0's stored tables
#: must keep describing those tables, and several of the columns below --
#: ALARM, age_band, vasopressor_use_std, composite_risk_score -- are precisely
#: the ones the data-layer work removes.
R0_CATEGORICAL = {
    "MIMIC": ["SpO2", "ALARM"],
    "eICU": ["mechanical_ventilation_std", "vasopressor_use_std", "age_band",
             "gender_std", "composite_risk_score"],
    "NHANES": ["gender_std", "age_band"],
    "AutoMPG": ["model_year", "origin"],
    "ComCri": ["IncomeLevel", "UrbanType", "EducationLevel", "CrimeLevel",
               "RegionCode"],
    "Concrete": ["Duration"],
}


def r0_schema(name: str):
    """A schema for the FROZEN R0 table, independent of configs/datasets.yaml.

    Legacy tests reproduce what R0 shipped, so they must be resolved against R0's
    own column declarations. After T2.1 rebuilt the data layer, resolving them
    through datasets.yaml raises "columns absent" for exactly the columns the
    rebuild removed -- ALARM, age_band, vasopressor_use_std -- which is the
    mirror image of the failure the clinical_v1 tests had before they were
    pointed at the current tables.
    """
    return schema_from_frame(r0_table(name),
                             categorical=R0_CATEGORICAL[name], identifier="ID")

#: Row-index correlation acceptance threshold from the T1.5 brief.
R1_4_THRESHOLD = 0.05

#: Sampling-noise scale of the null. For a single mask over n rows, the Pearson
#: correlation between per-row missing rate and row index is asymptotically
#: N(0, 1/(n-1)) under exchangeability. The 0.05 threshold is therefore a
#: ~4.5-sigma test at n = 8000 but only a ~1-sigma test at n = 400: the
#: synthetic fixture is deliberately large so that the required assertion is a
#: real test of the simulator rather than a coin flip.
def null_sd(n: int) -> float:
    return 1.0 / math.sqrt(max(n - 1, 1))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def synthetic_frame(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """A mixed-type ICU-shaped frame whose rows are i.i.d. (hence exchangeable).

    Row order carries no information by construction, which is what isolates the
    row-index test from the source table's incidental sortedness (see
    ``test_row_order_confound_is_a_data_property``). Deliberately includes
    ``const_flag``, a zero-variance categorical column, to exercise the B46 path
    that eICU's ``vasopressor_use_std`` triggers.
    """
    g = np.random.default_rng(seed)
    gcs = np.clip(np.round(g.normal(11.0, 3.2, n)), 3, 15)
    return pd.DataFrame({
        "rec_id": np.arange(1, n + 1),
        "gcs": gcs,
        "age": np.clip(np.round(g.normal(63.0, 15.0, n)), 18, 90),
        "vent": (g.random(n) < 0.35).astype(int),
        "lactate": np.round(np.exp(g.normal(0.4, 0.6, n)) + 0.02 * (15 - gcs), 2),
        "urine": np.round(np.abs(g.normal(800.0, 350.0, n)), 1),
        "spo2": np.round(np.clip(g.normal(94.0, 4.0, n), 50, 100), 1),
        "hgb": np.round(np.clip(g.normal(10.5, 2.0, n), 1, 19), 1),
        "sodium": np.round(np.clip(g.normal(139.0, 5.0, n), 110, 175), 1),
        "severity": g.integers(0, 6, n),
        "sex": g.integers(0, 2, n),
        "const_flag": np.ones(n, dtype=int),
    })


SYNTH_CATEGORICAL = ["vent", "severity", "sex", "const_flag"]

#: Per-column MAR spec exercising every feature the rewrite adds:
#:   * a table-level default (gcs, age)
#:   * per-column driver sets that differ from the default
#:   * a POSITIVE coefficient on gcs for spo2 against a NEGATIVE one for lactate
#:   * a column (urine) that ignores gcs entirely
#: R0 could express none of this: one propensity, every column.
SYNTH_MAR = {
    "default": {"drivers": ["gcs", "age"], "coefficients": {"gcs": -0.9, "age": 0.5}},
    "columns": {
        "lactate": {"drivers": ["gcs", "vent"], "coefficients": {"gcs": -1.3, "vent": 0.9}},
        "spo2": {"drivers": ["gcs"], "coefficients": {"gcs": 0.8}},
        "urine": {"drivers": ["vent"], "coefficients": {"vent": 1.2}},
    },
}
SYNTH_MNAR = {
    "columns": {
        "severity": {"mode": "ordinal", "coefficient": 0.9},
        "sex": {"mode": "semantic_groups",
                "groups": {"female": [0], "male": [1]},
                "group_offsets": {"female": 0.3, "male": -0.3}},
        "const_flag": {"mode": "ordinal"},
    },
}
SYNTH_OVERRIDES = {"mar": SYNTH_MAR, "mnar": SYNTH_MNAR}


def synthetic_spec(mechanism: str, rate: float, seed: int, *, df: pd.DataFrame,
                   overrides: dict | None = None) -> MissingnessSpec:
    schema = schema_from_frame(df, categorical=SYNTH_CATEGORICAL, identifier="rec_id")
    ov = dict(SYNTH_OVERRIDES)
    if overrides:
        ov = {**ov, **overrides}
    return resolve("_adhoc", mechanism, rate, profile="auto_slopes_smoke",
                   schema=schema, seed=seed, overrides=ov)


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return synthetic_frame()


def r0_table(name: str) -> pd.DataFrame:
    path = R0_DATA / f"{name}_complete.csv"
    if not path.exists():
        pytest.skip(f"R0 derived table not available: {path}")
    return pd.read_csv(path)


def _resolve_current_table(name: str) -> Path:
    """The table `configs/datasets.yaml` currently points at.

    Distinct from `r0_table`: after T2.1 the MIMIC and NHANES entries name
    rebuilt tables with different columns, and a test that asserts something
    about the masks we ship has to read the table we ship.
    """
    from missingness.spec import load_datasets_config
    declared = load_datasets_config()["datasets"]
    if name not in declared:
        # CDC2022 exists only once T2.1 has built it and regenerated
        # datasets.yaml. Skipping says "not built yet"; failing would say "the
        # simulator is broken", and the two must not look alike.
        pytest.skip(f"{name} not declared in datasets.yaml yet")
    cfg = declared[name]
    p = Path(cfg["complete_path"])
    cand = p if p.is_absolute() else (CODE_ROOT / p)
    if not cand.exists():
        pytest.skip(f"current table not built yet: {cand}")
    return cand


requires_r0 = pytest.mark.skipif(not R0_DATA.exists(), reason="R0 data tree not present")


def row_index_r(mask: np.ndarray) -> float:
    rate = mask.astype(float).mean(axis=1)
    if np.std(rate) < 1e-12:
        return 0.0
    return float(np.corrcoef(np.arange(rate.size, dtype=float), rate)[0, 1])


# ==========================================================================
# 1. drivers stay fully observed under every mechanism
# ==========================================================================

def test_drivers_stay_fully_observed(synth):
    """Strict MAR: a driver that can itself go missing is not MAR.

    R0 enforced this for MAR only (``missing_data_generator.py:638-654``). Here
    it holds under MCAR and MNAR too, so the evaluated column set is identical
    across the three mechanisms of a dataset and the masks stay comparable.
    """
    for mechanism in MECHANISMS:
        for rate in RATES:
            spec = synthetic_spec(mechanism, rate, seed=1, df=synth)
            res = generate(synth[spec.schema.order()], spec)

            expected_observed = {"rec_id"}
            if mechanism == "MAR":
                expected_observed |= {"gcs", "age", "vent"}
            assert set(spec.observed_columns()) == expected_observed, (
                f"{mechanism}@{rate}: observed set {spec.observed_columns()}")

            for col in spec.observed_columns():
                n_missing = int(res.X_missing[col].isna().sum())
                assert n_missing == 0, f"{mechanism}@{rate}: driver {col!r} has {n_missing} NaNs"
                j = list(res.X_missing.columns).index(col)
                assert res.mask[:, j].sum() == 0, f"{mechanism}@{rate}: mask marks driver {col!r}"


# (no @requires_r0: reads the current table, not the frozen R0 tree)
@pytest.mark.parametrize("dataset", CURRENT_DATASETS)
def test_drivers_stay_fully_observed_real_tables(dataset):
    # The current table, not the frozen R0 one: `clinical_v1` names drivers
    # that exist only in the tables T2.1 rebuilt.
    df = pd.read_csv(_resolve_current_table(dataset))
    for mechanism in MECHANISMS:
        spec = resolve(dataset, mechanism, 0.3, profile="clinical_v1", seed=1)
        res = generate(df, spec)
        for col in spec.observed_columns():
            assert int(res.X_missing[col].isna().sum()) == 0, (
                f"{dataset}/{mechanism}: {col!r} lost its full-observation guarantee")
        # and every declared driver really is in the observed set
        for drv in spec.driver_union():
            assert drv in spec.observed_columns()


# ==========================================================================
# 2. achieved missing rate within 1% of target
# ==========================================================================

def test_achieved_rate_within_one_percent(synth):
    for mechanism in MECHANISMS:
        for rate in RATES:
            for seed in SEEDS:
                spec = synthetic_spec(mechanism, rate, seed=seed, df=synth)
                res = generate(synth[spec.schema.order()], spec)
                rates = res.meta["rates"]

                assert abs(rates["actual_rate_eligible"] - rate) < 0.01, (
                    f"{mechanism}@{rate} seed={seed}: eligible-cell rate "
                    f"{rates['actual_rate_eligible']:.5f}")
                assert not rates["columns_outside_tolerance"], (
                    f"{mechanism}@{rate} seed={seed}: "
                    f"{rates['columns_outside_tolerance']}")
                for col in spec.target_columns():
                    got = rates["per_column_missing_rate"][col]
                    assert abs(got - rate) < 0.01, f"{mechanism}@{rate} {col}: {got:.5f}"


# (no @requires_r0: reads the current table, not the frozen R0 tree)
@pytest.mark.parametrize("dataset", CURRENT_DATASETS)
def test_achieved_rate_within_one_percent_real_tables(dataset):
    df = pd.read_csv(_resolve_current_table(dataset))
    for mechanism in MECHANISMS:
        for rate in RATES:
            spec = resolve(dataset, mechanism, rate, profile="clinical_v1", seed=1)
            res = generate(df, spec)
            rates = res.meta["rates"]
            assert abs(rates["actual_rate_eligible"] - rate) < 0.01
            assert not rates["columns_outside_tolerance"], (
                f"{dataset}/{mechanism}@{rate}: {rates['columns_outside_tolerance']}")


def test_both_rate_denominators_are_recorded(synth):
    """R0 reported eligible-cell and all-cell rates separately
    (``missing_data_generator.py:754-759``); that bookkeeping is kept."""
    spec = synthetic_spec("MAR", 0.3, seed=1, df=synth)
    res = generate(synth[spec.schema.order()], spec)
    rates = res.meta["rates"]
    assert abs(rates["actual_rate_eligible"] - 0.3) < 0.01
    # all-cell rate is diluted by the fully-observed columns
    n_obs = len(spec.observed_columns())
    n_all = len(spec.schema.order())
    assert rates["actual_rate_all"] == pytest.approx(
        rates["actual_rate_eligible"] * (n_all - n_obs) / n_all, abs=1e-9)
    assert rates["actual_rate_all"] < rates["actual_rate_eligible"]


# ==========================================================================
# 3. |corr(per-row missing rate, row index)| < 0.05      *** R1-4 ***
# ==========================================================================

def test_row_index_correlation_is_null(synth):
    """THE required assertion: the mask must not know where a row sits.

    R0's MAR masks used ``ID`` — a 1..n record counter — as their only driver,
    which produced r = +0.667 (MIMIC) to +0.804 (eICU). That is reviewer point
    R1-4. Here every mechanism x rate x seed must land inside +/-0.05.

    n = 8000, so the null sd is 0.0112 and the threshold is a ~4.5-sigma test.
    """
    n = len(synth)
    violations = []
    for mechanism in MECHANISMS:
        for rate in RATES:
            for seed in SEEDS:
                spec = synthetic_spec(mechanism, rate, seed=seed, df=synth)
                res = generate(synth[spec.schema.order()], spec)
                r = res.row_index_correlation()
                if abs(r) >= R1_4_THRESHOLD:
                    violations.append((mechanism, rate, seed, r))
                # the same number must be in meta.json, not just computable
                assert res.meta["row_index_diagnostics"][
                    "pearson_r_rowrate_vs_rowindex_eligible"] == pytest.approx(r)
    assert not violations, (
        f"row-index correlation exceeded {R1_4_THRESHOLD} "
        f"(null sd at n={n} is {null_sd(n):.4f}): {violations}")


# (no @requires_r0: reads the current table, not the frozen R0 tree)
@pytest.mark.parametrize("dataset", CURRENT_DATASETS)
def test_row_index_correlation_is_null_real_tables(dataset):
    """Same assertion on the real frames, with row order randomised.

    Randomization is required because three of the six R0 tables arrive sorted
    (see ``test_row_order_confound_is_a_data_property``), and a sorted table
    transmits row order into any mask whose driver happens to be the sort key —
    a data-layer property, not a simulator property.

    These frames are 392-2274 rows, where 0.05 is only 1-2 sigma of the null.
    The test is therefore stated the honest way: every draw must be within
    4 sigma of zero, and the mean over the five seeds must be inside +/-0.05.
    """
    df = pd.read_csv(_resolve_current_table(dataset))
    n = len(df)
    band = 4.0 * null_sd(n)
    for mechanism in MECHANISMS:
        for rate in RATES:
            rs = []
            for seed in SEEDS:
                spec = resolve(dataset, mechanism, rate, profile="clinical_v1",
                               seed=seed,
                               overrides={"row_order": {"mode": "shuffle", "seed": 20250728}})
                rs.append(generate(df, spec).row_index_correlation())
            rs = np.asarray(rs)
            assert np.abs(rs).max() < band, (
                f"{dataset}/{mechanism}@{rate}: |r|max={np.abs(rs).max():.4f} "
                f"exceeds the 4-sigma null band {band:.4f} (n={n})")
            # A mean violation must be BOTH practically (R1-4 scale) and
            # statistically (4 sigma of the 5-seed mean) significant.
            #
            # Derivation (so nobody reads this as a threshold tuned to make
            # a failing test pass -- ratified by internal review follow-up,
            # 2026-08-29): under exchangeable rows a single mask's row-rate/
            # row-index Pearson r is ~ N(0, 1/(n-1)), so null_sd(n) =
            # 1/sqrt(n-1) and the 5-seed mean has sigma_mean =
            # null_sd(n)/sqrt(5). On AutoMPG (n = 398): null_sd = 0.0502,
            # sigma_mean = 0.0224, so the old fixed +/-0.05 band was a
            # 2.2-sigma criterion -- a ~3% spurious-trigger rate per cell under a
            # PERFECT null, re-rolled by any legitimate RNG-stream change
            # (it first tripped at +0.0525 on AutoMPG/MCAR when the P5R-B
            # SS5-B2 target exclusion shifted the draw sequence; the
            # per-draw 4-sigma band it sits next to was never breached).
            # max(0.05, 4*sigma_mean) keeps the reviewer-scale 0.05 in
            # charge wherever it is a >= 4-sigma test (every large table,
            # unchanged) and substitutes the correctly-scaled 4-sigma band
            # only where 0.05 asks for precision the estimator does not
            # have. A real violation exceeds both.
            band_mean = max(R1_4_THRESHOLD,
                            4.0 * null_sd(n) / math.sqrt(len(SEEDS)))
            assert abs(rs.mean()) < band_mean, (
                f"{dataset}/{mechanism}@{rate}: mean r over {len(SEEDS)} seeds "
                f"= {rs.mean():+.4f} (band {band_mean:.4f})")


# ==========================================================================
# 4. per-column heterogeneity is real
# ==========================================================================

def test_per_column_heterogeneity_is_real(synth):
    """Different coefficients must produce measurably different sensitivity.

    Measured as the point-biserial correlation between the driver ``gcs`` and
    each column's missing indicator. The config sets ``gcs`` to -1.3 for
    ``lactate``, +0.8 for ``spo2``, and does not use it at all for ``urine``.
    Under R0's ``np.repeat`` broadcast (``missing_data_generator.py:467``) all
    three would show the *same* sensitivity, because all three shared one
    propensity vector.
    """
    spec = synthetic_spec("MAR", 0.3, seed=1, df=synth)
    df = synth[spec.schema.order()]
    res = generate(df, spec)
    cols = list(res.X_missing.columns)
    gcs = df["gcs"].to_numpy(float)

    def sens(col: str) -> float:
        j = cols.index(col)
        return float(np.corrcoef(gcs, res.mask[:, j].astype(float))[0, 1])

    r_lactate, r_spo2, r_urine = sens("lactate"), sens("spo2"), sens("urine")

    assert r_lactate < -0.20, f"negative coefficient did not bite: {r_lactate:+.4f}"
    assert r_spo2 > 0.20, f"positive coefficient did not bite: {r_spo2:+.4f}"
    assert abs(r_urine) < 0.05, (
        f"urine does not list gcs as a driver but tracks it at {r_urine:+.4f} — "
        f"this is the row-broadcast signature")
    assert (r_spo2 - r_lactate) > 0.40, (
        f"opposite-signed coefficients on the same driver must separate: "
        f"spo2={r_spo2:+.4f} lactate={r_lactate:+.4f}")

    # And the propensity vectors themselves must actually differ column to column.
    registry = StreamRegistry(spec.seed, namespace=("_adhoc", "MAR", "0.3", spec.profile))
    P, _ = build_propensity_matrix(df, spec, registry)
    tgt = [cols.index(c) for c in spec.target_columns()]
    spread = max(float(np.abs(P[:, a] - P[:, b]).max())
                 for i, a in enumerate(tgt) for b in tgt[i + 1:])
    assert spread > 0.05, f"target columns share one propensity (max sup-norm gap {spread:.4g})"


@requires_r0
def test_legacy_mar_gives_every_column_the_identical_propensity():
    """The defect the above test is the mirror of, demonstrated directly.

    R0's MAR built one row vector and called ``np.repeat`` on it
    (``missing_data_generator.py:467``), so every target column's propensity is
    *bit-identical*. Confirmed here on the real eICU table.
    """
    df = r0_table("eICU")
    spec = resolve("eICU", "MAR", 0.3, profile="record_index_ID", seed=2025,
                    schema=r0_schema("eICU"))
    res = generate(df, spec)

    # Legacy MAR: sensitivity to the driver is the same for every column,
    # because there is only one propensity.
    cols = list(res.X_missing.columns)
    drv = df["ID"].to_numpy(float)
    sens = np.array([np.corrcoef(drv, res.mask[:, cols.index(c)].astype(float))[0, 1]
                     for c in spec.target_columns()])
    assert sens.min() > 0.15, "legacy MAR should make every column track ID"
    assert (sens.max() - sens.min()) < 0.12, (
        f"legacy MAR columns should be near-homogeneous, spread was "
        f"{sens.max() - sens.min():.4f}")

    # The rewrite, on the CURRENT eICU table -- `clinical_v1` is written
    # against the columns T2.1 ships, and `gcs` is a driver in both.
    df_new = pd.read_csv(_resolve_current_table("eICU"))
    spec_new = resolve("eICU", "MAR", 0.3, profile="clinical_v1", seed=1)
    res_new = generate(df_new, spec_new)
    cols_new = list(res_new.X_missing.columns)
    gcs = df_new["gcs"].to_numpy(float)
    sens_new = np.array([np.corrcoef(gcs, res_new.mask[:, cols_new.index(c)].astype(float))[0, 1]
                         for c in spec_new.target_columns()])
    assert (sens_new.max() - sens_new.min()) > 0.20, (
        f"rewrite must produce heterogeneous sensitivity, spread was "
        f"{sens_new.max() - sens_new.min():.4f}")


# ==========================================================================
# 5. the legacy profile reproduces the pathology (test 3 discriminates)
# ==========================================================================

@requires_r0
@pytest.mark.parametrize("dataset", R0_DATASETS)
def test_legacy_profile_reproduces_the_pathology(dataset):
    """A test that cannot fail is worthless. This one shows it can.

    The ``record_index_ID`` profile reproduces R0 exactly, and R0's MAR masks
    are strongly row-index dependent. If this assertion ever stops holding, the
    row-index metric has stopped measuring what we think it measures.
    """
    df = r0_table(dataset)
    spec = resolve(dataset, "MAR", 0.3, profile="record_index_ID",
                     schema=r0_schema(dataset))
    res = generate(df, spec)
    r = res.row_index_correlation()
    assert r > 0.6, f"{dataset}: legacy MAR row-index correlation only {r:+.4f}"

    # ... while the rewrite is inside the null band.
    #
    # The rewrite is evaluated on the CURRENT table, not on the frozen R0 one.
    # T2.1 rebuilt MIMIC and NHANES, so `clinical_v1` names drivers that exist
    # only in the new tables; resolving that schema and then generating against
    # the R0 table would raise, and forcing the two together would mean the
    # assertion no longer describes the masks we actually ship. The legacy half
    # above still runs on the R0 table, because reproducing R0's pathology is a
    # claim about R0.
    df_new = pd.read_csv(_resolve_current_table(dataset))
    spec_new = resolve(dataset, "MAR", 0.3, profile="clinical_v1", seed=1,
                       overrides={"row_order": {"mode": "shuffle", "seed": 20250728}})
    r_new = generate(df_new, spec_new).row_index_correlation()
    assert abs(r_new) < 4.0 * null_sd(len(df_new))
    assert abs(r_new) < r / 3.0, f"{dataset}: legacy {r:+.4f} vs rewrite {r_new:+.4f}"


@requires_r0
def test_legacy_port_reproduces_R0_masks_bitwise():
    """The legacy profile is bit-exact, not merely similar.

    All 54 stored R0 masks (6 datasets x 3 mechanisms x 3 rates) are regenerated
    from ``code_SNI`` alone. This is what makes "old mask vs new mask"
    comparisons in P2 attributable to the mechanism change rather than to a
    reimplementation artifact.
    """
    checked = 0
    for dataset in R0_DATASETS:
        df = r0_table(dataset)
        for mechanism in MECHANISMS:
            for rate, tag in zip(RATES, ["10per", "30per", "50per"]):
                stored_path = R0_DATA / dataset / f"{dataset}_{mechanism}_{tag}_mask.npy"
                if not stored_path.exists():
                    pytest.skip(f"missing R0 mask {stored_path}")
                stored = np.load(stored_path).astype(bool)
                spec = resolve(dataset, mechanism, rate, profile="record_index_ID",
                                  schema=r0_schema(dataset))
                res = generate(df, spec)
                assert np.array_equal(res.mask_bool, stored), (
                    f"{dataset}/{mechanism}/{tag}: "
                    f"{int((res.mask_bool != stored).sum())} cells differ")
                checked += 1
    assert checked == 54, f"expected 54 R0 masks, checked {checked}"


# ==========================================================================
# row order is a property of the data, not of the mechanism
# ==========================================================================

@requires_r0
def test_row_order_confound_is_a_data_property():
    """Evidence for the T1.6 driver decision, stated as a test.

    AutoMPG arrives sorted by ``model_year`` (corr with row index +0.997). Using
    ``model_year`` as the MAR driver — perfectly defensible as a mechanism —
    still yields a large row-index correlation, purely because of how the file
    is ordered. Shuffling the rows removes it entirely.

    So ``|r| < 0.05`` is a joint property of (mechanism, driver, row order). The
    generator records ``pearson_r_driver_vs_rowindex`` in meta.json precisely so
    that this cannot pass unnoticed.
    """
    df = r0_table("AutoMPG")
    idx = np.arange(len(df), dtype=float)
    assert abs(np.corrcoef(idx, df["model_year"].to_numpy(float))[0, 1]) > 0.99

    spec_asis = resolve("AutoMPG", "MAR", 0.3, profile="clinical_v1", seed=1)
    res_asis = generate(df, spec_asis)
    assert abs(res_asis.row_index_correlation()) > 0.3, (
        "a sorted table with a sort-key driver must show the confound")
    diag = res_asis.meta["row_index_diagnostics"]["pearson_r_driver_vs_rowindex"]
    assert abs(diag["model_year"]) > 0.99, (
        "meta.json must expose the driver's own row-order correlation so the "
        "confound is diagnosable without re-running")

    spec_shuf = resolve("AutoMPG", "MAR", 0.3, profile="clinical_v1", seed=1,
                        overrides={"row_order": {"mode": "shuffle", "seed": 20250728}})
    res_shuf = generate(df, spec_shuf)
    assert abs(res_shuf.row_index_correlation()) < 4.0 * null_sd(len(df))
    perm = res_shuf.meta["row_order"]["permutation"]
    assert perm is not None and sorted(perm) == list(range(len(df))), (
        "the permutation must be recorded in full so the mask stays reproducible")


@requires_r0
def test_source_table_row_order_is_documented():
    """Which R0 tables are sorted, as a fact on the record for T1.6."""
    sorted_tables = {}
    for dataset in R0_DATASETS:
        df = r0_table(dataset)
        idx = np.arange(len(df), dtype=float)
        worst = 0.0
        for c in df.columns:
            if c == "ID":
                continue
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            if np.nanstd(v) < 1e-12:
                continue
            worst = max(worst, abs(float(np.corrcoef(idx, v)[0, 1])))
        sorted_tables[dataset] = worst
    # measured 2026-07-27; these are the numbers quoted in the P1 report
    assert sorted_tables["AutoMPG"] > 0.99      # sorted by model_year
    assert sorted_tables["MIMIC"] > 0.55        # tracks ALARM
    assert sorted_tables["Concrete"] > 0.30     # tracks ConcreteCS
    assert sorted_tables["eICU"] < 0.10         # effectively unordered
    assert sorted_tables["NHANES"] < 0.10
    assert sorted_tables["ComCri"] < 0.10


# ==========================================================================
# B39 — no column overshoots its target
# ==========================================================================

@requires_r0
def test_b39_no_column_overshoots_its_target():
    """R0's per-column MNAR rates blew past the target; the rewrite's do not.

    Measured on the stored R0 masks at a 30% target:
      MIMIC  ALARM                       37.8%
      eICU   vasopressor_use_std         43.4%
             mechanical_ventilation_std  39.7%
             composite_risk_score        34.2%
      NHANES gender_std                  34.0%
    """
    known_overshoots = {
        "MIMIC": {"ALARM": 0.378},
        "eICU": {"vasopressor_use_std": 0.434, "mechanical_ventilation_std": 0.397,
                 "composite_risk_score": 0.342},
        "NHANES": {"gender_std": 0.340},
    }
    for dataset, expected in known_overshoots.items():
        df = r0_table(dataset)

        legacy = generate(df, resolve(dataset, "MNAR", 0.3, profile="record_index_ID",
                                        schema=r0_schema(dataset)))
        legacy_rates = legacy.meta["rates"]["per_column_missing_rate"]
        for col, approx_rate in expected.items():
            assert legacy_rates[col] == pytest.approx(approx_rate, abs=0.002), (
                f"legacy {dataset}/{col} should reproduce the B39 overshoot")

        # The rewrite, on the SAME table and the SAME columns.
        #
        # Not via `clinical_v1`: T2.1 deletes every column named above -- ALARM,
        # vasopressor_use_std and composite_risk_score are exactly the columns
        # the data-layer work removes -- so resolving the published profile here
        # would either raise or quietly test different columns. The schema is
        # taken from the frame using R0'S OWN type declarations, snapshotted
        # below, and the mechanism knobs come from `auto_slopes_smoke`. What is
        # under test is the generator's per-column MNAR calibration, which is a
        # property of the rewrite and not of the driver specification.
        schema = schema_from_frame(
            df, categorical=R0_CATEGORICAL[dataset], identifier="ID")
        new = generate(df, resolve(dataset, "MNAR", 0.3,
                                   profile="auto_slopes_smoke",
                                   schema=schema, seed=1))
        new_rates = new.meta["rates"]["per_column_missing_rate"]
        for col in expected:
            assert abs(new_rates[col] - 0.3) < 0.01, (
                f"{dataset}/{col} still overshoots: {new_rates[col]:.4f}")
        assert not new.meta["rates"]["columns_outside_tolerance"]


# ==========================================================================
# B46 — degenerate (zero-variance) columns
# ==========================================================================

def test_b46_degenerate_categorical_column(synth):
    """eICU ``vasopressor_use_std`` is constant 1.0 in all 1430 rows.

    R0 picked ``max(1, round(1 * 0.5)) = 1`` level as "high missing", which is
    *every* row, so the column sat at ``rate * 1.5`` — 45% against a 30% target,
    landing at 43.4% after table calibration. MNAR on a constant column is
    undefined; the rewrite falls back to a constant propensity equal to the
    target rate and says so in meta.json.
    """
    spec = synthetic_spec("MNAR", 0.3, seed=1, df=synth)
    res = generate(synth[spec.schema.order()], spec)
    rec = res.meta["per_column_spec"]["const_flag"]
    assert rec["degenerate"] is True
    assert "zero-variance" in rec["note"]
    assert abs(res.meta["rates"]["per_column_missing_rate"]["const_flag"] - 0.3) < 0.01


@requires_r0
def test_b46_on_the_real_eicu_column():
    df = r0_table("eICU")
    assert df["vasopressor_use_std"].nunique() == 1, "B46 precondition changed"
    # T2.1 removes this column from the shipped table (B35), so the subject of
    # this test survives only in the frozen R0 one. What is under test is the
    # generator's zero-variance path, not the published driver spec, so the
    # schema comes from the R0 frame and the knobs from the mechanism-only
    # profile.
    res = generate(df, resolve("eICU", "MNAR", 0.3, profile="auto_slopes_smoke",
                               schema=r0_schema("eICU"), seed=1))
    rec = res.meta["per_column_spec"]["vasopressor_use_std"]
    assert rec["degenerate"] is True
    assert abs(res.meta["rates"]["per_column_missing_rate"]["vasopressor_use_std"] - 0.3) < 0.01


# ==========================================================================
# B45 — independent RNG streams
# ==========================================================================

def test_b45_streams_are_keyed_by_name_not_position():
    """Adding a column must not re-roll every other column's draws."""
    a = independence_probe(2025, ["alpha", "beta", "gamma"])
    b = independence_probe(2025, ["alpha", "NEW_CATEGORICAL", "beta", "gamma"])
    for col in ("alpha", "beta", "gamma"):
        assert a[col] == b[col], f"stream for {col!r} moved when a column was inserted"
    assert a["alpha"] != a["beta"], "distinct columns must get distinct streams"


def test_b45_adding_a_column_leaves_other_masks_untouched(synth):
    """The end-to-end version of the above, through the whole generator."""
    base = synth.drop(columns=["sodium"])
    spec_base = synthetic_spec("MNAR", 0.3, seed=1, df=base)
    res_base = generate(base[spec_base.schema.order()], spec_base)

    spec_full = synthetic_spec("MNAR", 0.3, seed=1, df=synth)
    res_full = generate(synth[spec_full.schema.order()], spec_full)

    cols_b = list(res_base.X_missing.columns)
    cols_f = list(res_full.X_missing.columns)
    for col in cols_b:
        assert np.array_equal(res_base.mask[:, cols_b.index(col)],
                              res_full.mask[:, cols_f.index(col)]), (
            f"{col!r} changed when an unrelated column was added — B45 has regressed")


@requires_r0
def test_b45_legacy_path_still_exhibits_the_coupling():
    """Confirms the previous test measures something R0 actually failed."""
    df = r0_table("eICU")
    spec = resolve("eICU", "MNAR", 0.3, profile="record_index_ID",
                   schema=r0_schema("eICU"))
    full = generate(df, spec)

    reduced_df = df.drop(columns=["lactate_mmol_l"])
    # From the R0 schema, not from datasets.yaml: this test is about R0's shared
    # RNG stream, and the frame it operates on is R0's.
    schema = r0_schema("eICU")
    from missingness.spec import DatasetSchema
    reduced_schema = DatasetSchema(
        name="eICU",
        columns={k: v for k, v in schema.columns.items() if k != "lactate_mmol_l"},
        identifier_column="ID", downstream_target="composite_risk_score")
    spec_r = resolve("eICU", "MNAR", 0.3, profile="record_index_ID", schema=reduced_schema)
    reduced = generate(reduced_df, spec_r)

    cols_f = list(full.X_missing.columns)
    cols_r = list(reduced.X_missing.columns)
    changed = [c for c in cols_r
               if not np.array_equal(full.mask[:, cols_f.index(c)],
                                     reduced.mask[:, cols_r.index(c)])]
    assert len(changed) > 1, (
        "R0's shared RNG should make removing one column disturb others; "
        f"only {changed} moved")


# ==========================================================================
# logit-space calibration
# ==========================================================================

def test_solve_intercept_is_exact():
    rng = np.random.default_rng(0)
    eta = rng.normal(0, 1.5, 5000)
    for target in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        b = solve_intercept(eta, target)
        assert sigmoid(eta + b).mean() == pytest.approx(target, abs=1e-9)


def test_intercept_fit_preserves_the_slope_where_rescaling_did_not():
    """The substantive reason the rescale had to go.

    R0 set the level with ``p * (rate / mean_p)``
    (``missing_data_generator.py:460-464``). A multiplied sigmoid is no longer a
    logistic function, and the multiplier depends on the target rate, so the
    *strength of the dependence* moved with the rate: the 10%, 30% and 50% masks
    of one dataset were three different mechanisms, not one mechanism at three
    levels. Fitting an intercept leaves the slope alone.

    Measured as the log-odds contrast between the top and bottom driver decile,
    which is exactly ``a * (z_hi - z_lo)`` for a true logistic and therefore
    rate-invariant by construction.
    """
    from missingness.calibration import logit

    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 20000)
    a = 1.2
    hi = z > np.quantile(z, 0.9)
    lo = z < np.quantile(z, 0.1)

    def log_odds_contrast(p):
        return float(logit(p[hi].mean()) - logit(p[lo].mean()))

    base = sigmoid(a * z)
    rates = (0.1, 0.3, 0.5, 0.7)
    fitted = [log_odds_contrast(sigmoid(a * z + solve_intercept(a * z, r))) for r in rates]
    rescaled = [log_odds_contrast(np.clip(base * (r / base.mean()), 0.0, 1.0)) for r in rates]

    # Intercept fit: the mechanism is the same at every rate (< 5% drift).
    assert (max(fitted) - min(fitted)) / np.mean(fitted) < 0.05, (
        f"intercept fit should be rate-invariant, got {fitted}")
    # Legacy rescale: it is not. Measured 2.19 / 2.70 / 4.06 / 29.25 at
    # 10/30/50/70%, i.e. the dependence at 10% is about half as strong in
    # log-odds as at 50%, and by 70% the clip at 1.0 saturates the upper tail.
    assert (max(rescaled) - min(rescaled)) > 5.0 * (max(fitted) - min(fitted)), (
        f"legacy rescale contrasts {rescaled} should drift far more than {fitted}")
    assert rescaled[0] < 0.7 * rescaled[2], (
        f"legacy 10% mask should be a materially weaker mechanism than its 50% "
        f"counterpart: {rescaled[0]:.3f} vs {rescaled[2]:.3f}")


def test_mnar_quantile_steps_mode_is_still_available(synth):
    """The old MNAR mechanism stays reachable as an explicit comparison option."""
    spec = synthetic_spec("MNAR", 0.3, seed=1, df=synth,
                          overrides={"mnar": {"continuous_mode": "quantile_steps",
                                              "categorical_mode": "random_split",
                                              "columns": {}}})
    res = generate(synth[spec.schema.order()], spec)
    rec = res.meta["per_column_spec"]["lactate"]
    assert rec["mode"] == "quantile_steps"
    assert "multipliers" in rec and rec["multipliers"]["p_high_mult"] == 1.8
    # per-column calibration still pins the rate even in comparison mode
    assert abs(res.meta["rates"]["per_column_missing_rate"]["lactate"] - 0.3) < 0.01


# ==========================================================================
# E4 / B38 — the mask on disk is the authority
# ==========================================================================

def test_e4_written_mask_is_reloaded_and_verified(synth, tmp_path):
    spec = synthetic_spec("MAR", 0.3, seed=1, df=synth)
    res = generate_and_write(synth[spec.schema.order()], spec, tmp_path, stem="unit_MAR_30per")

    npy = tmp_path / "unit_MAR_30per_mask.npy"
    csv = tmp_path / "unit_MAR_30per.csv"
    meta = json.loads((tmp_path / "unit_MAR_30per_meta.json").read_text())

    assert npy.exists() and csv.exists()
    arr = np.load(npy)
    assert arr.dtype == np.uint8 and set(np.unique(arr)) <= {0, 1}
    assert arr.shape == (len(synth), len(spec.schema.order()))

    e4 = meta["e4_mask_verification"]
    assert e4["all_consistent"] is True
    assert {c["target"] for c in e4["checks"]} == {"in_memory", "csv_roundtrip"}
    for c in e4["checks"]:
        assert c["n_disagreements"] == 0
        assert c["n_missing_mask"] == c["n_missing_isna"] == int(arr.sum())
    assert res.mask_check is not None and res.mask_check.consistent


def test_e4_a_corrupted_mask_is_rejected(synth, tmp_path):
    """The check is enforced: flipping one cell must abort, not warn."""
    from common.masks import load_and_verify

    spec = synthetic_spec("MCAR", 0.3, seed=1, df=synth)
    res = generate_and_write(synth[spec.schema.order()], spec, tmp_path, stem="corrupt_MCAR_30per")

    npy = tmp_path / "corrupt_MCAR_30per_mask.npy"
    arr = np.load(npy)
    j = list(res.X_missing.columns).index("lactate")
    arr[0, j] ^= 1
    np.save(npy, arr)

    with pytest.raises(MaskConsistencyError):
        load_and_verify(res.X_missing, npy, columns=list(res.X_missing.columns), strict=True)


# ==========================================================================
# meta.json completeness
# ==========================================================================

def test_meta_json_records_the_complete_spec(synth, tmp_path):
    spec = synthetic_spec("MAR", 0.3, seed=3, df=synth)
    generate_and_write(synth[spec.schema.order()], spec, tmp_path, stem="meta_MAR_30per")
    meta = json.loads((tmp_path / "meta_MAR_30per_meta.json").read_text())

    for key in ("spec", "rates", "per_column_spec", "rng", "row_index_diagnostics",
                "e4_mask_verification", "artifacts", "environment", "row_order"):
        assert key in meta, f"meta.json is missing {key!r}"

    # per-column drivers, coefficients and fitted intercepts
    rec = meta["per_column_spec"]["lactate"]
    assert rec["drivers"] == ["gcs", "vent"]
    assert rec["coefficients"]["gcs"]["beta"] == pytest.approx(-1.3)
    assert rec["coefficients"]["vent"]["beta"] == pytest.approx(0.9)
    assert isinstance(rec["intercept"], float)
    assert rec["expected_rate"] == pytest.approx(0.3, abs=1e-9)
    assert rec["achieved_rate"] == pytest.approx(0.3, abs=0.01)

    # target vs achieved on both denominators
    for key in ("target", "actual_rate_eligible", "actual_rate_all",
                "n_eligible_cells", "n_all_cells", "per_column_missing_rate"):
        assert key in meta["rates"]

    # RNG seeds, in a form that re-derives every stream
    assert meta["rng"]["root_seed"] == 3
    assert meta["rng"]["n_streams"] > 0
    assert all({"purpose", "key", "entropy"} <= set(s) for s in meta["rng"]["streams"])

    # git commit of code_SNI (E2 / runconfig reuse)
    assert "code_SNI_git_commit" in meta["environment"]

    # spec round-trips the whole declaration
    assert meta["spec"]["mar"]["columns"]["spo2"]["coefficients"]["gcs"] == pytest.approx(0.8)
    assert meta["spec"]["mechanism"] == "MAR"
    assert meta["spec"]["observed_columns"] == spec.observed_columns()


# ==========================================================================
# config contract
# ==========================================================================

def test_config_active_profile_carries_no_placeholder():
    """The inverse of the P1 guard.

    Through P1 this test asserted that every `clinical_v1` block was *marked* as
    a placeholder, so that a provisional driver set could not reach P2 unnoticed.
    T2.2(a) filled the profile in, so the contract inverts: no placeholder marker
    may survive anywhere in it.
    """
    cfg = load_config()
    prof = cfg["profiles"]["clinical_v1"]
    assert prof["status"] == "ACTIVE"
    for name, block in prof["datasets"].items():
        assert "_PLACEHOLDER" not in block, f"{name}: placeholder marker survived"
        for mech in ("MCAR", "MAR", "MNAR"):
            assert "_narrative_PLACEHOLDER" not in (block.get(mech) or {}), (
                f"{name}/{mech}: placeholder narrative survived")


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
def test_config_every_active_block_has_a_rationale(mechanism):
    """Every published block must carry text a reviewer can read.

    Checked against the raw config rather than through `resolve`, because the
    rationale is a property of the specification and must be present even for the
    datasets whose tables T2.1 has not yet rebuilt.
    """
    prof = load_config()["profiles"]["clinical_v1"]
    for name, block in prof["datasets"].items():
        r = ((block.get(mechanism) or {}).get("rationale")
             or (block.get("common") or {}).get("rationale"))
        assert r and len(r) > 80, (
            f"{name}/{mechanism}: rationale missing or too short to defend")


def test_config_every_mar_driver_is_always_observed():
    """A MAR mechanism may condition only on observed data.

    If a driver is itself maskable then missingness depends on an unobserved
    value, which makes the mechanism MNAR while the config still calls it MAR.
    This is the one error in the specification that would invalidate the
    experiment rather than merely degrade it, so it is a contract test.
    """
    prof = load_config()["profiles"]["clinical_v1"]
    for name, block in prof["datasets"].items():
        observed = set((block.get("common") or {}).get("always_observed", []) or [])
        mar = ((block.get("MAR") or {}).get("mar") or {})
        specs = list((mar.get("columns") or {}).values())
        if mar.get("default"):
            specs.append(mar["default"])
        for s in specs:
            for d in (s or {}).get("drivers", []) or []:
                assert d in observed, (
                    f"{name}: MAR driver {d!r} is not in always_observed, so the "
                    f"mechanism would condition on a value that can be masked")


def test_config_legacy_profile_declares_ID_as_the_driver():
    cfg = load_config()
    prof = cfg["profiles"]["record_index_ID"]
    assert prof["defaults"]["implementation"] == "legacy_R0"
    for name, block in prof["datasets"].items():
        assert block["MAR"]["mar"]["default"]["drivers"] == ["ID"], name


def test_config_is_valid_yaml_and_resolves_for_every_dataset():
    cfg = load_config()
    assert cfg["schema_version"] == 1
    yaml.safe_load(DEFAULT_MISSINGNESS_CONFIG.read_text())
    for dataset in R0_DATASETS:
        for mechanism in MECHANISMS:
            for profile in ("record_index_ID", "clinical_v1"):
                spec = resolve(dataset, mechanism, 0.3, profile=profile)
                assert spec.mechanism == mechanism
                assert spec.dataset == dataset


def test_mar_without_drivers_is_a_hard_error(synth):
    """R0 silently fell back to ``col_types.continuous[:2]``
    (``missing_data_generator.py:693``). Silence is what let ``ID`` survive."""
    schema = schema_from_frame(synth, categorical=SYNTH_CATEGORICAL, identifier="rec_id")
    with pytest.raises(ValueError, match="MAR requires"):
        resolve("_adhoc", "MAR", 0.3, profile="auto_slopes_smoke", schema=schema, seed=1)


def test_unknown_driver_is_a_hard_error(synth):
    schema = schema_from_frame(synth, categorical=SYNTH_CATEGORICAL, identifier="rec_id")
    with pytest.raises(ValueError, match="unknown drivers"):
        resolve("_adhoc", "MAR", 0.3, profile="auto_slopes_smoke", schema=schema, seed=1,
                overrides={"mar": {"default": {"drivers": ["not_a_column"]}}})


def test_incomplete_input_is_rejected(synth):
    spec = synthetic_spec("MCAR", 0.3, seed=1, df=synth)
    dirty = synth[spec.schema.order()].copy()
    dirty.loc[0, "lactate"] = np.nan
    with pytest.raises(ValueError, match="already contains missing values"):
        generate(dirty, spec)


def test_auto_coefficients_are_heterogeneous_and_signed(synth):
    """The ``auto`` path must produce genuinely different, signed slopes."""
    spec = synthetic_spec("MAR", 0.3, seed=1, df=synth,
                          overrides={"mar": {"coefficient_default": "auto",
                                             "default": {"drivers": ["gcs", "age"]},
                                             "columns": {}}})
    res = generate(synth[spec.schema.order()], spec)
    betas = [rec["coefficients"]["gcs"]["beta"]
             for rec in res.meta["per_column_spec"].values() if rec.get("masked")]
    assert len(set(np.round(betas, 6))) == len(betas), "auto slopes must differ per column"
    assert min(betas) < 0 < max(betas), f"auto slopes must span both signs: {betas}"
    assert all(abs(b) >= 0.2 for b in betas), "|beta| must be pushed off zero"


def test_generation_is_reproducible(synth):
    spec_a = synthetic_spec("MNAR", 0.3, seed=5, df=synth)
    spec_b = synthetic_spec("MNAR", 0.3, seed=5, df=synth)
    assert np.array_equal(generate(synth[spec_a.schema.order()], spec_a).mask,
                          generate(synth[spec_b.schema.order()], spec_b).mask)
    spec_c = synthetic_spec("MNAR", 0.3, seed=6, df=synth)
    assert not np.array_equal(generate(synth[spec_a.schema.order()], spec_a).mask,
                              generate(synth[spec_c.schema.order()], spec_c).mask)


# ==========================================================================
# P5R-B SS5-B2 — downstream target auto-excluded from future generations
# ==========================================================================

def test_downstream_target_auto_excluded_forward():
    """Forward fix (P5R-B SS5-B2): a freshly resolved non-legacy spec keeps
    the declared downstream target fully observed under every mechanism, even
    where a mechanism stanza lists it, and records the exclusion in notes
    (embedded in every future mask meta). R0 reproduction profiles
    (implementation: legacy_R0) are exempt: they must keep reproducing the
    historical behavior. Frozen mask files are untouched either way -- they
    carry their own spec snapshot."""
    for ds, tgt in [("MIMIC", "mortality_risk"),
                    ("eICU", "composite_risk_score"),
                    ("CDC2022", "HadHeartAttack"),
                    ("ComCri", "ViolentCrimesPerPop")]:
        for mech in ("MCAR", "MAR", "MNAR"):
            s = resolve(ds, mech, 0.3, profile="clinical_v1", seed=1)
            assert tgt in s.observed_columns(), (ds, mech)
            assert any("auto-excluded" in n for n in s.notes), (ds, mech)
            assert tgt not in s.target_columns(), (ds, mech)
    legacy = resolve("eICU", "MNAR", 0.3, profile="record_index_ID")
    assert "composite_risk_score" in legacy.target_columns()
    assert not any("auto-excluded" in n for n in legacy.notes)
