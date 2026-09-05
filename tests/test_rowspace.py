"""Tests for the row-space assertion (P2b decision 3).

This is the guard against the one failure mode in the R1 pipeline that produces
no error at all: pairing a shuffled mask with an unshuffled table. Shapes match,
dtypes match, every metric computes, and every number is wrong. A guard against a
silent failure is worth nothing unless it is itself tested, so each test below
constructs the mistake and checks that it is caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.rowspace import (
    RowSpace,
    RowSpaceMismatch,
    assert_same_rowspace,
    audit_tree,
    fingerprint,
)

CODE_ROOT = Path(__file__).resolve().parent.parent


def _frame(ids):
    return pd.DataFrame({"ID": list(ids), "x": np.arange(len(ids), dtype=float)})


def test_fingerprint_is_order_sensitive_and_content_stable():
    a = _frame([1, 2, 3, 4])
    assert fingerprint(a.ID.tolist()) == fingerprint(a.ID.tolist())
    # A permutation must change it -- that is the whole point.
    assert fingerprint([1, 2, 3, 4]) != fingerprint([1, 3, 2, 4])
    # ...and it must not depend on which other columns are present.
    b = a.assign(extra=1)
    assert (RowSpace.of_frame(a, "a").digest == RowSpace.of_frame(b, "b").digest)


def test_fingerprint_survives_a_csv_round_trip(tmp_path: Path):
    """The tables ship as CSV, so a dtype round-trip must not move the digest."""
    a = _frame([10, 20, 30])
    p = tmp_path / "t.csv"
    a.to_csv(p, index=False)
    assert RowSpace.of_frame(a, "mem").digest == RowSpace.of_frame(pd.read_csv(p), "csv").digest


def test_shuffled_table_paired_with_unshuffled_mask_is_rejected():
    """The actual mistake this module exists to prevent."""
    table = _frame(range(1, 51))
    shuffled = table.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(RowSpaceMismatch, match="row-space mismatch"):
        assert_same_rowspace(RowSpace.of_frame(table, "table"),
                             RowSpace.of_frame(shuffled, "mask"))


def test_matching_rowspaces_pass_and_return_the_digest():
    table = _frame(range(1, 21))
    d = assert_same_rowspace(RowSpace.of_frame(table, "table"),
                             RowSpace.of_frame(table.copy(), "mask"),
                             RowSpace.of_frame(table.copy(), "target"))
    assert d == fingerprint(table.ID.tolist())


def test_unverifiable_artifact_is_not_treated_as_verified():
    """A mask with no fingerprint cannot be shown to match. 'Cannot be shown to
    match' must not read the same as 'matches' -- that is how silent errors get
    blessed."""
    table = _frame(range(1, 11))
    nofp = RowSpace("legacy_mask", None)
    with pytest.raises(RowSpaceMismatch, match="cannot verify"):
        assert_same_rowspace(RowSpace.of_frame(table, "table"), nofp)
    # Opting out is possible, but only explicitly.
    assert assert_same_rowspace(RowSpace.of_frame(table, "table"), nofp,
                                strict=False)


def test_frame_without_identifier_reports_unverifiable():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert RowSpace.of_frame(df, "no_id").digest is None


def test_of_mask_meta_accepts_both_shape_encodings():
    """`shape` is a dict in the current generator and a list in older metadata."""
    for shape, n in (({"n_rows": 7, "n_cols": 3}, 7), ([7, 3], 7), (None, None)):
        rs = RowSpace.of_mask_meta({"shape": shape,
                                    "row_order": {"rowspace_digest": "abc"}}, "m")
        assert rs.digest == "abc" and rs.n_rows == n


def test_audit_tree_flags_a_mismatched_pair(tmp_path: Path):
    tables, masks = tmp_path / "tables", tmp_path / "masks" / "DS"
    tables.mkdir(parents=True)
    masks.mkdir(parents=True)
    table = _frame(range(1, 31))
    table.to_csv(tables / "DS_complete.csv", index=False)

    good = {"shape": {"n_rows": 30},
            "row_order": {"rowspace_digest": fingerprint(table.ID.tolist())}}
    bad = {"shape": {"n_rows": 30},
           "row_order": {"rowspace_digest": fingerprint(list(reversed(table.ID.tolist())))}}
    (masks / "DS_MAR_30per_meta.json").write_text(json.dumps(good))
    (masks / "DS_MCAR_30per_meta.json").write_text(json.dumps(bad))

    out = audit_tree(tables, tmp_path / "masks", ["DS"])
    status = dict(zip(out["mask"], out["status"]))
    assert status["DS_MAR_30per_meta"] == "ok"
    assert status["DS_MCAR_30per_meta"].startswith("MISMATCH")


@pytest.mark.skipif(not (CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled").exists(),
                    reason="shipped masks not generated yet")
def test_shipped_shuffled_tree_is_internally_consistent():
    """The masks we actually ship, against the tables we actually ship."""
    import yaml
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    out = audit_tree(CODE_ROOT / "data" / "derived_shuffled",
                     CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled",
                     list(cfg["datasets"]))
    bad = out[out.status != "ok"]
    assert len(out) > 0
    assert bad.empty, bad.to_string(index=False)


@pytest.mark.skipif(not (CODE_ROOT / "data" / "derived" / "MIMIC_complete.csv").exists()
                    or not (CODE_ROOT / "data" / "derived_shuffled" / "MIMIC_complete.csv").exists(),
                    reason="both table variants required")
def test_the_two_shipped_variants_are_distinguishable():
    """If these ever collided, the guard would be vacuous."""
    a = RowSpace.of_frame(pd.read_csv(CODE_ROOT / "data/derived/MIMIC_complete.csv"), "as_is")
    b = RowSpace.of_frame(pd.read_csv(CODE_ROOT / "data/derived_shuffled/MIMIC_complete.csv"), "shuf")
    assert a.digest != b.digest
