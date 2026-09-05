"""Tests for the grid aggregation (P2b decision 1).

This module decides how the numbers in Table 1 are produced, so its two rules
need to be tested rather than assumed:

* the median is the primary aggregate for **every** method and metric -- the
  point being that no method gets a special rule;
* every method carries a divergence rate, because two baselines fall below the
  column-mean line and the current presentation hides both (B70, B73).

The coverage check is tested too: an undisclosed coverage gap is finding B3, the
most dangerous defect in R0 that the reviewers did *not* find.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats.aggregate_grid import (
    REFERENCE_METHODS,
    aggregate,
    coverage,
    divergence_table,
    load_runs,
)


def _runs(**over) -> pd.DataFrame:
    """Five seeds of one cell, with TabCSDI's real NHANES R^2 values."""
    base = dict(dataset="NHANES", mechanism="MAR", rate=0.3, method="TabCSDI")
    r2 = over.pop("r2", [0.0067, 0.1610, -19.345, -223.995, -8.702])
    rows = [{**base, **over, "seed": s, "cont_R2": v, "cont_NRMSE": 0.1 + i * 0.01,
             "runtime_sec": 100 + i}
            for i, (s, v) in enumerate(zip([1, 2, 3, 5, 8], r2))]
    return pd.DataFrame(rows)


def test_median_is_reported_and_differs_from_the_mean_where_it_matters():
    """The case the rule exists for: a mean over these five seeds is -50.4 and
    describes no run that happened."""
    agg = aggregate(_runs())
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["cont_R2_median"] == pytest.approx(-8.702)
    assert row["cont_R2_mean"] == pytest.approx(-50.3749, abs=1e-3)
    # Both are emitted -- the ESM shows mean+/-sd alongside.
    assert row["cont_R2_sd"] > 90


def test_median_rule_applies_to_every_method_not_just_the_diverging_one():
    good = _runs(method="MissForest", r2=[0.48, 0.49, 0.47, 0.50, 0.48])
    agg = aggregate(pd.concat([_runs(), good], ignore_index=True))
    assert set(agg.method) == {"TabCSDI", "MissForest"}
    # Every row has a median column; none is special-cased away.
    assert agg["cont_R2_median"].notna().all()
    assert agg["cont_R2_mean"].notna().all()


def test_divergence_rate_counts_runs_below_the_column_mean():
    agg = aggregate(_runs())
    div = divergence_table(agg)
    row = div.iloc[0]
    assert row.n_runs == 5
    assert row.n_R2_negative == 3          # -19.3, -224.0, -8.7
    assert row.divergence_rate == pytest.approx(0.6)
    assert row.worst_R2 == pytest.approx(-223.995)


def test_gain_style_total_divergence_is_visible():
    """B73: GAIN is below the line on every dataset. The column must say so."""
    gain = _runs(method="GAIN", r2=[-1.1, -1.5, -2.6, -3.5, -1.9])
    div = divergence_table(aggregate(gain))
    assert div.iloc[0].divergence_rate == 1.0
    assert not div.iloc[0].is_reference_method


def test_meanmode_is_flagged_as_the_reference_not_as_a_failure():
    """MeanMode's rate is ~1 by construction -- R^2 uses the true values'
    variance on masked cells while mean imputation fills from observed ones. It
    must not read as the same event as GAIN scoring below the same line."""
    mm = _runs(method="MeanMode", r2=[-0.001, -0.002, -0.001, -0.003, -0.001])
    div = divergence_table(aggregate(mm))
    assert div.iloc[0].divergence_rate == 1.0
    assert div.iloc[0].is_reference_method
    assert "MeanMode" in REFERENCE_METHODS


def test_coverage_reports_an_incomplete_cell():
    short = _runs().iloc[:3]                # only three seeds
    cov = coverage(short, expected_seeds=5)
    assert len(cov) == 1
    assert cov.iloc[0].n_seeds == 3
    assert not bool(cov.iloc[0].complete)


def test_coverage_passes_a_complete_cell():
    cov = coverage(_runs(), expected_seeds=5)
    assert bool(cov.iloc[0].complete)


def test_load_runs_reports_unreadable_files_instead_of_dropping_them(tmp_path: Path,
                                                                    capsys):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "metrics_summary.json").write_text(
        json.dumps({"dataset": "D", "mechanism": "MAR", "rate": 0.3,
                    "method": "M", "seed": 1, "cont_R2": 0.5}))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "metrics_summary.json").write_text("{not json")

    df = load_runs(tmp_path)
    assert len(df) == 1
    assert "WARNING" in capsys.readouterr().out


def test_mask_consistency_detects_a_cell_run_against_two_masks():
    """The P2b failure, as a test.

    A mask regeneration landed 7 minutes into a running experiment: same table,
    same row order, different driver set. The row-space digest could not see it
    -- that guard is about row *order* -- and nothing else looked. Half the arms
    used a different input from the other half.
    """
    from stats.aggregate_grid import mask_consistency

    df = pd.DataFrame([
        dict(dataset="ok", mechanism="MAR", rate=0.3, seed=s, mask_md5="aaa")
        for s in (1, 2, 3)
    ] + [
        dict(dataset="mixed", mechanism="MAR", rate=0.3, seed=1, mask_md5="aaa"),
        dict(dataset="mixed", mechanism="MAR", rate=0.3, seed=2, mask_md5="bbb"),
    ])
    mc = mask_consistency(df).set_index("dataset")
    assert bool(mc.loc["ok", "consistent"])
    assert not bool(mc.loc["mixed", "consistent"])
    assert mc.loc["mixed", "n_distinct"] == 2


def test_mask_consistency_is_silent_when_the_column_is_absent():
    """Older runs predate the hash; absence must not masquerade as agreement."""
    from stats.aggregate_grid import mask_consistency
    assert mask_consistency(_runs()).empty
