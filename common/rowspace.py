"""P2b decision 3: one global assertion that table, mask and target share a row space.

The generator permutes rows when `row_order: shuffle` and does **not** invert the
permutation, so a shuffled mask indexes a shuffled table. Pair a shuffled mask
with an unshuffled table and nothing raises: shapes match, dtypes match, every
metric computes, and every number is wrong. That is the single most likely place
for a silent error in the whole R1 pipeline, which is why the P2b instruction
asks for an assertion rather than a convention.

The mechanism is a **row-space fingerprint**: a hash of the identifier column in
the order it appears. Two artifacts belong together exactly when their
fingerprints agree. It is cheap, needs no metadata, and cannot be satisfied by
accident.

Usage at every point where the three meet::

    from common.rowspace import RowSpace, assert_same_rowspace

    assert_same_rowspace(
        RowSpace.of_frame(complete, "table"),
        RowSpace.of_mask_meta(meta, "mask"),
        RowSpace.of_frame(downstream, "target"),
    )

`RowSpace.of_mask_meta` reads the fingerprint the generator stamped into
`meta.json`; older masks that predate the stamp report `None` and are reported as
unverifiable rather than silently accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


class RowSpaceMismatch(AssertionError):
    """Raised when two artifacts do not index the same rows in the same order."""


def fingerprint(values: Sequence) -> str:
    """Stable 16-hex digest of a row ordering.

    Uses the identifier values in order, so it is invariant to which columns are
    present and to dtype round-trips through CSV, but changes under any
    permutation.
    """
    h = hashlib.sha256()
    for v in values:
        h.update(repr(v).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class RowSpace:
    label: str
    digest: Optional[str]
    n_rows: Optional[int] = None
    source: Optional[str] = None

    @classmethod
    def of_frame(cls, df: pd.DataFrame, label: str,
                 identifier: str = "ID", source: Optional[str] = None) -> "RowSpace":
        if identifier not in df.columns:
            # No identifier: fall back to positional row count only, and say so.
            return cls(label, None, len(df), source)
        return cls(label, fingerprint(df[identifier].tolist()), len(df), source)

    @classmethod
    def of_mask_meta(cls, meta: dict, label: str,
                     source: Optional[str] = None) -> "RowSpace":
        ro = (meta or {}).get("row_order") or {}
        shape = (meta or {}).get("shape")
        if isinstance(shape, dict):
            n = shape.get("n_rows")
        elif isinstance(shape, (list, tuple)) and shape:
            n = shape[0]
        else:
            n = None
        return cls(label, ro.get("rowspace_digest"), n, source)

    @classmethod
    def of_mask_path(cls, meta_path: Path, label: str) -> "RowSpace":
        return cls.of_mask_meta(json.loads(Path(meta_path).read_text()), label,
                                str(meta_path))


def assert_same_rowspace(*spaces: RowSpace, strict: bool = True) -> str:
    """Raise unless every artifact shares one row space.

    `strict` (the default) treats an unverifiable artifact -- one with no
    fingerprint -- as a failure. That is deliberate: a mask written before the
    fingerprint existed cannot be shown to match, and "cannot be shown to match"
    must not read the same as "matches".
    """
    named = list(spaces)
    if len(named) < 2:
        raise ValueError("assert_same_rowspace needs at least two artifacts")

    missing = [s for s in named if s.digest is None]
    if missing and strict:
        raise RowSpaceMismatch(
            "cannot verify row space for: "
            + ", ".join(f"{s.label}({s.source or 'no fingerprint'})" for s in missing)
            + ". Regenerate the artifact so it carries a fingerprint, or pass "
              "strict=False and accept that the pairing is unchecked.")

    checkable = [s for s in named if s.digest is not None]
    digests = {s.digest for s in checkable}
    if len(digests) > 1:
        detail = "; ".join(f"{s.label}={s.digest} (n={s.n_rows})" for s in checkable)
        raise RowSpaceMismatch(
            "row-space mismatch -- these artifacts do not index the same rows in "
            "the same order, so any metric computed from them would be silently "
            f"wrong: {detail}")

    n = {s.n_rows for s in checkable if s.n_rows is not None}
    if len(n) > 1:
        raise RowSpaceMismatch(f"row counts disagree: {sorted(n)}")
    return checkable[0].digest


def audit_tree(table_dir: Path, mask_dir: Path,
               datasets: Iterable[str], identifier: str = "ID") -> pd.DataFrame:
    """Check every (table, mask) pair under two directories. Used by the grid
    launcher as a pre-flight, so a mismatch stops the run before 55 GPU-hours."""
    rows = []
    for ds in datasets:
        tp = Path(table_dir) / f"{ds}_complete.csv"
        if not tp.exists():
            rows.append({"dataset": ds, "status": f"table missing: {tp}"})
            continue
        tbl = RowSpace.of_frame(pd.read_csv(tp), f"{ds}:table", identifier, str(tp))
        for mp in sorted((Path(mask_dir) / ds).glob("*_meta.json")):
            msk = RowSpace.of_mask_path(mp, f"{ds}:{mp.stem}")
            try:
                assert_same_rowspace(tbl, msk)
                status = "ok"
            except RowSpaceMismatch as exc:
                status = f"MISMATCH: {exc}"[:200]
            rows.append({"dataset": ds, "mask": mp.stem, "status": status,
                         "table_digest": tbl.digest, "mask_digest": msk.digest})
    return pd.DataFrame(rows)
