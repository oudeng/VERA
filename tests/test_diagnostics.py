"""Regression tests for the six-item mask diagnostic.

This module is the evidence for reviewer point R1-4, so a silent failure in it
is worse than no diagnostic at all -- it would certify masks nobody checked.
Four such failures were found by hand while validating it against AutoMPG, and
each has a test here:

* item 3 read `per_column_spec[c]["abs_rate_error"]`, a key the generator never
  writes, so every rate error came back NaN;
* `targets` read `meta["target_columns"]` instead of `meta["spec"]["target
  _columns"]`, so the always-observed drivers were judged against a target rate
  they are meant never to reach;
* item 1's fixed 0.05 threshold is a 1-sigma test at n = 392 and a 4.5-sigma
  test at n = 8000, so a null mask fails it a third of the time on the small
  tables;
* item 5's spread has to be read against n, since the statistic's sampling floor
  grows as 1/sqrt(n).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missingness import diagnostics as D


def _bundle(mask: pd.DataFrame, table: pd.DataFrame, meta: dict,
            rate: float = 0.3) -> D.MaskBundle:
    return D.MaskBundle("T", "MAR", rate, mask, table, meta)


def test_null_sigma_is_the_exchangeable_standard_deviation():
    for n in (392, 2274, 8000):
        assert D.null_sigma(n) == pytest.approx(1.0 / np.sqrt(n - 1))
    # The point of reporting it: the same threshold is a very different test.
    assert 0.05 / D.null_sigma(392) < 1.5
    assert 0.05 / D.null_sigma(8000) > 4.0


def test_row_index_correlation_detects_an_ordered_mask():
    n = 2000
    idx = np.arange(n)
    # R0's pathology, reproduced: 7 % missing at the top of the file rising to
    # 49 % at the bottom. Ten columns, because the statistic is a mean over
    # columns and the real tables have 8-20 of them; with only three, a row rate
    # can take four values and the attenuation drags a genuine +0.67 down to
    # +0.46, which would be a property of the test rather than of the mask.
    cols = [f"c{i}" for i in range(10)]
    p = 0.07 + 0.42 * idx / n
    rng = np.random.RandomState(0)
    m = pd.DataFrame({c: rng.rand(n) < p for c in cols})
    r = D.row_index_correlation(m, cols)
    assert r > 0.6, f"an ordered mask must be detected, got {r}"

    flat = pd.DataFrame({c: rng.rand(n) < 0.3 for c in cols})
    assert abs(D.row_index_correlation(flat, cols)) < 0.05


def test_logodds_contrast_is_stable_across_target_rates():
    """A fixed coefficient must give a fixed contrast when only the rate moves.

    This is what B49 broke: R0's mean-linear rescale returned 2.19 / 2.70 / 4.06
    at rates 0.1 / 0.3 / 0.5, a 63 % spread, so its three rates were three
    mechanisms rather than three strengths of one.
    """
    from scipy.optimize import brentq
    rng = np.random.RandomState(0)
    n, beta = 20000, 1.5
    x = rng.normal(size=n)
    got = []
    for target in (0.1, 0.3, 0.5):
        b = brentq(lambda b: (1 / (1 + np.exp(-(beta * x + b)))).mean() - target,
                   -20, 20)
        p = 1 / (1 + np.exp(-(beta * x + b)))
        mask = pd.DataFrame({"c": rng.rand(n) < p})
        got.append(D.logodds_contrast(mask, pd.DataFrame({"d": x}), "d", ["c"]))
    spread = (max(got) - min(got)) / abs(np.mean(got))
    assert spread < 0.15, f"contrast should be rate-invariant, spread {spread:.3f}"


def test_driver_sensitivity_recovers_an_intended_sign_flip():
    """Item 4. R0 could not express this at all -- one propensity vector was
    broadcast to every column (B50), so every row of this matrix was identical.
    """
    rng = np.random.RandomState(1)
    n = 4000
    x = rng.normal(size=n)
    up = 1 / (1 + np.exp(-(1.2 * x)))       # more missing as the driver rises
    down = 1 / (1 + np.exp(-(-1.2 * x)))    # ...and less, in another column
    mask = pd.DataFrame({"up": rng.rand(n) < up, "down": rng.rand(n) < down})
    s = D.driver_sensitivity(mask, pd.DataFrame({"d": x}), ["d"], ["up", "down"])
    a = float(s.set_index("column").loc["up", "d"])
    b = float(s.set_index("column").loc["down", "d"])
    assert a > 0.2 and b < -0.2, f"sign flip not recovered: up={a}, down={b}"


def _calibrated_column(n: int, rate: float, rng) -> np.ndarray:
    """Exactly round(rate * n) missing cells, as the generator's per-column
    calibration produces. A raw Bernoulli draw would not do: at n = 1000 and
    p = 0.3 its sampling sd is 0.0145, so it misses the 0.01 tolerance about half
    the time and the test would be measuring binomial noise."""
    k = int(round(rate * n))
    out = np.zeros(n, dtype=bool)
    out[rng.choice(n, size=k, replace=False)] = True
    return out


def test_item3_ignores_the_always_observed_drivers():
    """The bug: a driver is never masked, so |0 - rate| = rate, and item 3 read
    that as the worst rate error in the table."""
    n = 1000
    rng = np.random.RandomState(2)
    table = pd.DataFrame({"ID": np.arange(1, n + 1),
                          "drv": rng.normal(size=n),
                          "a": rng.normal(size=n), "b": rng.normal(size=n)})
    mask = pd.DataFrame({"ID": np.zeros(n, bool),
                         "drv": np.zeros(n, bool),
                         "a": _calibrated_column(n, 0.3, rng),
                         "b": _calibrated_column(n, 0.3, rng)})
    meta = {"spec": {"target_columns": ["a", "b"],
                     "observed_columns": ["ID", "drv"],
                     "mar": {"driver_union": ["drv"]}},
            "rates": {"columns_outside_tolerance": []}}
    out = D.diagnose_bundle(_bundle(mask, table, meta))
    assert out["n_target_columns"] == 2
    # Without the fix this is 0.3 -- the full target rate, contributed by `drv`.
    assert out["max_abs_rate_error"] <= D.RATE_TOLERANCE, out["max_abs_rate_error"]
    assert out["n_columns_out_of_tolerance"] == 0
    assert out["rate_pass"] is True
    assert out["drivers"] == ["drv"]
    assert out["drivers_fully_observed"] is True
    assert out["rate_verdict_agrees_with_generator"] is True


def test_item3_still_fires_when_a_column_really_misses_its_target():
    n = 1000
    rng = np.random.RandomState(3)
    table = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    mask = pd.DataFrame({"a": _calibrated_column(n, 0.30, rng),
                         "b": _calibrated_column(n, 0.42, rng)})
    meta = {"spec": {"target_columns": ["a", "b"]},
            "rates": {"columns_outside_tolerance": ["b"]}}
    out = D.diagnose_bundle(_bundle(mask, table, meta))
    assert out["max_abs_rate_error"] == pytest.approx(0.12, abs=1e-6)
    assert out["n_columns_out_of_tolerance"] == 1
    assert out["rate_pass"] is False
    assert out["rate_verdict_agrees_with_generator"] is True


def test_noise_adjusted_pass_is_looser_only_where_n_is_small():
    """|r| = 0.051 is a real signal at n = 8000 and noise at n = 392."""
    def _out(n, slope):
        rng = np.random.RandomState(4)
        idx = np.arange(n)
        cols = [f"c{i}" for i in range(10)]
        base = np.clip(0.3 + slope * (idx - idx.mean()) / idx.std(), 0.01, 0.99)
        mask = pd.DataFrame({c: rng.rand(n) < base for c in cols})
        meta = {"spec": {"target_columns": cols}}
        table = pd.DataFrame({c: rng.normal(size=n) for c in cols})
        return D.diagnose_bundle(_bundle(mask, table, meta))

    # A null mask on the smallest table. |r| lands near the 0.05 line, which at
    # n = 392 is only 1 sigma, so the fixed criterion is a coin flip here while
    # the noise-adjusted one is not.
    small = _out(392, 0.0)
    assert small["row_index_sigmas"] < 3.0
    assert small["row_index_pass_noise_adjusted"] is True

    # A mask that really does encode row order, on a table large enough that the
    # fixed threshold is a 4.5-sigma test. Both criteria must reject it.
    big = _out(8000, 0.25)
    assert big["row_index_sigmas"] > 3.0
    assert big["row_index_pass"] is False
    assert big["row_index_pass_noise_adjusted"] is False


def test_load_bundle_accepts_both_the_nested_and_flat_layouts(tmp_path: Path):
    n = 50
    table = pd.DataFrame({"ID": np.arange(1, n + 1),
                          "a": np.arange(n, dtype=float)})
    meta = {"spec": {"target_columns": ["a"]}}
    arr = np.zeros((n, 2), dtype=np.uint8)
    arr[:10, 1] = 1

    for base in (tmp_path / "nested" / "DS", tmp_path / "flat"):
        base.mkdir(parents=True)
        (base / "DS_MAR_30per_meta.json").write_text(json.dumps(meta))
        np.save(base / "DS_MAR_30per_mask.npy", arr)
    tpath = tmp_path / "t.csv"
    table.to_csv(tpath, index=False)

    for root in (tmp_path / "nested", tmp_path / "flat"):
        b = D.load_bundle(root, "DS", "MAR", 0.3, tpath)
        assert b.mask.shape == (n, 2)
        assert b.targets == ["a"]

    with pytest.raises(FileNotFoundError):
        D.load_bundle(tmp_path / "nested", "DS", "MCAR", 0.3, tpath)
