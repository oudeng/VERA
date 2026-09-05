"""Mask loading with a hard consistency assertion (engineering principle E4).

P0 finding B38: R0 generated one ``*_mask.npy`` per (dataset, mechanism, rate)
and its documentation promised that every imputer consumed the same cached mask.
In fact **no code path ever loaded a .npy**. Both consumers
(``run_experiment.py:43-49`` and ``run_manifest_parallel.py:198-199``) read CSV
only, no manifest carried a ``mask_file`` column, and every run therefore fell
through to ``mask_df = X_missing.isna()``. The masks happened to agree because
all methods read the same missing-value CSV, but nothing checked that, and the
released artifact was never validated against the data it claimed to describe.

Here the ``.npy`` is the authority. It is loaded, aligned to the dataframe's
columns, and compared cell-by-cell with ``X_missing.isna()``. A mismatch raises
:class:`MaskConsistencyError` rather than being silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class MaskConsistencyError(RuntimeError):
    """Raised when the stored mask disagrees with the missing-value table."""


@dataclass(frozen=True)
class MaskCheck:
    """Outcome of a consistency check, suitable for logging into run_config."""

    mask_path: Optional[str]
    n_rows: int
    n_cols: int
    n_missing_mask: int
    n_missing_isna: int
    n_disagreements: int
    columns_checked: List[str]

    @property
    def consistent(self) -> bool:
        return self.n_disagreements == 0


def load_mask_npy(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load a ``.npy`` mask and label its columns.

    The stored array covers *all* columns of the generated table, including the
    identifier column that is excluded from masking. ``columns`` must therefore be
    the dataframe's full column list in its original order.
    """
    arr = np.load(Path(path))
    if arr.ndim != 2:
        raise MaskConsistencyError(f"{path}: expected a 2-D mask, got shape {arr.shape}")
    if arr.shape[1] != len(columns):
        raise MaskConsistencyError(
            f"{path}: mask has {arr.shape[1]} columns but {len(columns)} names were given "
            f"({list(columns)[:5]}...). The mask must be loaded with the FULL column list "
            f"of the generated table, identifier column included."
        )
    return pd.DataFrame(arr.astype(bool), columns=list(columns))


def load_and_verify(
    X_missing: pd.DataFrame,
    mask_path: Optional[Path],
    *,
    columns: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, MaskCheck]:
    """Return the evaluation mask together with the result of the E4 check.

    Parameters
    ----------
    X_missing
        Table with NaNs at the masked positions.
    mask_path
        Path to the cached ``.npy``. If ``None``, the mask is derived from
        ``X_missing.isna()`` and the check is recorded as vacuous — callers should
        treat that as a degraded mode and say so in the run config.
    columns
        Full column list of the generated table (identifier included). Defaults
        to ``X_missing.columns``, which is correct when the identifier is still
        present in ``X_missing``.
    strict
        Raise on disagreement (default). When False the disagreement count is
        returned and the ``.npy`` still wins, so a caller can quarantine the
        dataset instead of aborting a whole sweep.
    """
    isna = X_missing.isna()

    if mask_path is None:
        chk = MaskCheck(
            mask_path=None,
            n_rows=int(isna.shape[0]),
            n_cols=int(isna.shape[1]),
            n_missing_mask=int(isna.to_numpy().sum()),
            n_missing_isna=int(isna.to_numpy().sum()),
            n_disagreements=0,
            columns_checked=list(isna.columns),
        )
        return isna, chk

    full_cols = list(columns) if columns is not None else list(X_missing.columns)
    mask_full = load_mask_npy(Path(mask_path), full_cols)

    shared = [c for c in X_missing.columns if c in mask_full.columns]
    missing_names = [c for c in X_missing.columns if c not in mask_full.columns]
    if missing_names:
        raise MaskConsistencyError(
            f"{mask_path}: columns {missing_names} are present in the data but absent "
            f"from the stored mask."
        )

    mask = mask_full[shared].copy()
    mask.index = X_missing.index

    disagree = (mask[shared].to_numpy() != isna[shared].to_numpy())
    n_bad = int(disagree.sum())

    chk = MaskCheck(
        mask_path=str(mask_path),
        n_rows=int(mask.shape[0]),
        n_cols=len(shared),
        n_missing_mask=int(mask[shared].to_numpy().sum()),
        n_missing_isna=int(isna[shared].to_numpy().sum()),
        n_disagreements=n_bad,
        columns_checked=shared,
    )

    if n_bad and strict:
        rows, cols = np.where(disagree)
        sample = [(int(r), shared[c]) for r, c in zip(rows[:5], cols[:5])]
        raise MaskConsistencyError(
            f"{mask_path}: cached mask disagrees with X_missing.isna() at {n_bad} cells "
            f"(mask says {chk.n_missing_mask} missing, isna says {chk.n_missing_isna}). "
            f"First disagreements (row, column): {sample}. "
            f"Refusing to proceed — the released mask and the released table must agree."
        )

    return mask, chk
