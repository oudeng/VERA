"""The T2d.1 verdict must fail when it should, not only pass when it should.

A decision procedure that has only ever been exercised on favorable data is not
a decision procedure. Two of the cases below are the ones that matter:

  * a mean under the threshold hiding one dataset far over it -- the rule says
    "every one of the three", and averaging would have let CDC2022-style outliers
    through;
  * a batch that looks fast because the jobs which never finished were not
    counted -- that is exactly how the GPU queue produced `slowdown_vs_conc1`
    values of 5.0 while completing nothing at all.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pandas as pd
import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent

C1_REFS = [("MIMIC", 1952.0), ("NHANES", 1895.7), ("Concrete", 123.5)]


@pytest.fixture
def verdict(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "t2d1_verdict", CODE_ROOT / "tests" / "t2d1_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = tmp_path
    return mod


def _c1(ratios):
    return pd.DataFrame([
        {"dataset": d, "wall_sec": r * ref, "cuda_reference_sec": ref,
         "cont_R2": 0.5, "cont_NRMSE": 0.1}
        for (d, ref), r in zip(C1_REFS, ratios)])


def _c2(dataset, solo, wide, complete, n=12):
    return pd.DataFrame([
        {"dataset": dataset, "concurrency": 1, "sec_per_cell": solo,
         "all_completed": True},
        *[{"dataset": dataset, "concurrency": n, "sec_per_cell": wide,
           "all_completed": complete} for _ in range(n)]])


@pytest.mark.parametrize("ratios,expected", [
    ([0.8, 0.9, 0.85], True),
    ([3.0, 3.0, 3.0], True),        # the boundary is inclusive per the rule
    ([0.8, 3.5, 0.85], False),
    ([0.4, 4.0, 0.4], False),       # mean 1.6 -- must still fail
])
def test_c1_is_per_dataset_not_on_the_mean(verdict, ratios, expected):
    _c1(ratios).to_csv(verdict.OUT / "C1_single_cpu.csv", index=False)
    got, _ = verdict.c1()
    assert got is expected


def test_c1_missing_dataset_fails_rather_than_passes(verdict):
    _c1([0.8, 0.9, 0.85]).iloc[:2].to_csv(
        verdict.OUT / "C1_single_cpu.csv", index=False)
    got, rows = verdict.c1()
    assert got is False
    assert any("MISSING" in str(r) for r in rows)


@pytest.mark.parametrize("solo,wide,complete,expected", [
    (100, 120, True, True),
    (100, 150, True, True),         # exactly 1.5x, inclusive
    (100, 160, True, False),
    (100, 110, False, False),       # fast-looking, but a job never finished
])
def test_c2_counts_only_complete_batches(verdict, solo, wide, complete, expected):
    _c2("MIMIC", solo, wide, complete).to_csv(
        verdict.OUT / "C2_conc_mimic.csv", index=False)
    got, _ = verdict.c2()
    assert got is expected


def test_absent_data_reports_unmeasured_not_pass(verdict):
    for fn in (verdict.c1, verdict.c2, verdict.c3):
        got, _ = fn()
        assert got is None, "no data must never read as PASS"


def test_c3_degenerate_spread_uses_absolute_fallback(verdict):
    # All three seeds identical: without the fallback the tolerance would be 0
    # and C3 could never pass, making it unfalsifiable in the other direction.
    pd.DataFrame([{"dataset": "MIMIC", "wall_sec": 1.0, "cuda_reference_sec": 1.0,
                   "cont_R2": 0.500_02, "cont_NRMSE": 0.100_00}]).to_csv(
        verdict.OUT / "C1_single_cpu.csv", index=False)
    pd.DataFrame([
        {"dataset": "MIMIC", "seed": s, "early_stopping_disabled": True,
         "cont_R2": 0.5, "cont_NRMSE": 0.1} for s in (1, 2, 3)]).to_csv(
        verdict.OUT / "protocol_pairs.csv", index=False)
    got, rows = verdict.c3()
    assert any("abs fallback" in str(r) for r in rows)
    assert got is False     # MIMIC present but NHANES/Concrete absent
