"""Within-fold imputation protocol (T1.7, part 1) -- the machinery for R1-3.

What went wrong in R0
---------------------
The NHANES downstream path imputed the **whole table** and only afterwards cut
it into train / test::

    ext1/scripts/exp2_downstream_task_validation.py:391-404   generate missingness on the full table
    ext1/scripts/exp2_downstream_task_validation.py:460       X_imp = imputer.impute(X_missing=X_missing, ...)     # whole table
    ext1/scripts/exp2_downstream_task_validation.py:477       X_imp = baseline.impute(X_complete_cast, X_missing)  # whole table
    ext1/scripts/exp2_downstream_task_validation.py:517-520   X_train = X_imp.iloc[train_idx]; X_test = X_imp.iloc[test_idx]

Every imputer therefore saw the test rows while filling the training rows, which
is exactly reviewer point R1-3 ("imputation must be fitted entirely within the
training fold").  Line ``:477`` additionally hands ``X_complete_cast`` -- the
ground truth -- to the baseline, which is the separate oracle leak (B6) handled
by T1.4.

What this module provides
-------------------------
Two protocols, and a way to prove which one was used:

``PROTOCOL_FIT_TRANSFORM``
    The correct inductive protocol.  ``fit`` sees the training fold only;
    ``transform`` is then applied to the training fold and, separately, to the
    test fold.  Available for MeanMode / KNN / MICE / MissForest / HyperImpute /
    SNI.

``PROTOCOL_FOLD_INDEPENDENT``
    The documented fallback for genuinely transductive methods (GAIN / MIWAE /
    TabCSDI), which learn from the very matrix they complete and cannot expose a
    reusable ``transform``.  The train fold and the test fold are imputed by
    **two independently constructed imputer instances**, so no information flows
    between folds in either direction.  This is weaker than fit/transform (the
    test fold is still completed transductively *within itself*) but it removes
    the train/test leak, and the asymmetry is reported rather than hidden.

Neither protocol ever receives ``X_complete``.  Ground truth is used only by the
scorer, after imputation.

This module builds the protocol.  The downstream re-run itself is P2 work.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PROTOCOL_FIT_TRANSFORM",
    "PROTOCOL_FOLD_INDEPENDENT",
    "R0_LEAKAGE_EVIDENCE",
    "FoldSpec",
    "FoldImputation",
    "ProtocolReport",
    "detect_protocol",
    "make_holdout_folds",
    "make_kfold_folds",
    "impute_within_fold",
    "run_protocol",
    "verify_fold_independence",
]

PROTOCOL_FIT_TRANSFORM = "fit_transform"
PROTOCOL_FOLD_INDEPENDENT = "fold_independent"

#: Citable evidence for the response letter.
R0_LEAKAGE_EVIDENCE: Dict[str, str] = {
    "path": "ext1/scripts/exp2_downstream_task_validation.py",
    "generate_missingness": "391-404",
    "sni_whole_table_impute": "460",
    "baseline_whole_table_impute": "477",
    "split_after_impute": "517-520",
    "summary": (
        "The NHANES downstream panel imputed the full table before splitting, so "
        "test rows informed the training-fold imputation."
    ),
}


# --------------------------------------------------------------------------- #
# Fold plumbing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldSpec:
    """Positional (not label-based) row indices for one fold."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        tr = np.asarray(self.train_idx, dtype=int)
        te = np.asarray(self.test_idx, dtype=int)
        object.__setattr__(self, "train_idx", tr)
        object.__setattr__(self, "test_idx", te)
        overlap = np.intersect1d(tr, te)
        if overlap.size:
            raise ValueError(f"fold {self.fold_id}: train/test overlap on {overlap.size} rows")


def make_holdout_folds(
    n_rows: int,
    *,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: Optional[Sequence] = None,
) -> List[FoldSpec]:
    """A single stratified (or plain) hold-out split, expressed as one fold."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(int(n_rows))
    strat = None if stratify is None else np.asarray(stratify)
    train_idx, test_idx = train_test_split(
        idx, test_size=float(test_size), random_state=int(seed), stratify=strat
    )
    return [FoldSpec(0, np.sort(train_idx), np.sort(test_idx))]


def make_kfold_folds(
    n_rows: int,
    *,
    n_splits: int = 5,
    seed: int = 0,
    stratify: Optional[Sequence] = None,
) -> List[FoldSpec]:
    """K-fold (stratified when labels are given) fold specifications."""
    from sklearn.model_selection import KFold, StratifiedKFold

    idx = np.arange(int(n_rows))
    if stratify is None:
        splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        pairs = splitter.split(idx)
    else:
        splitter = StratifiedKFold(
            n_splits=int(n_splits), shuffle=True, random_state=int(seed)
        )
        pairs = splitter.split(idx, np.asarray(stratify))
    return [FoldSpec(i, tr, te) for i, (tr, te) in enumerate(pairs)]


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #


def detect_protocol(imputer: Any) -> str:
    """Return the protocol an imputer object supports.

    An imputer qualifies as inductive when it exposes callable ``fit`` and
    ``transform``.  Anything else falls back to fold-independent imputation.
    Methods may override the verdict with a class attribute
    ``supports_fit_transform: bool`` -- useful for wrappers that technically
    define ``transform`` but re-fit inside it.
    """
    override = getattr(imputer, "supports_fit_transform", None)
    if override is not None:
        return PROTOCOL_FIT_TRANSFORM if bool(override) else PROTOCOL_FOLD_INDEPENDENT
    has_fit = callable(getattr(imputer, "fit", None))
    has_transform = callable(getattr(imputer, "transform", None))
    return PROTOCOL_FIT_TRANSFORM if (has_fit and has_transform) else PROTOCOL_FOLD_INDEPENDENT


def _call_impute(imputer: Any, X_missing: pd.DataFrame) -> pd.DataFrame:
    """Call an imputer's one-shot entry point without ever passing ground truth."""
    if callable(getattr(imputer, "impute", None)):
        try:
            return imputer.impute(X_missing)
        except TypeError:
            # Leak-free keyword form used by the R1 SNI port.
            return imputer.impute(X_missing=X_missing, X_complete=None)
    if callable(getattr(imputer, "fit_transform", None)):
        return imputer.fit_transform(X_missing)
    raise TypeError(
        f"{type(imputer).__name__} exposes neither impute() nor fit_transform()"
    )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class FoldImputation:
    """Imputed train / test blocks for one fold, plus provenance."""

    fold_id: int
    protocol: str
    train_imputed: pd.DataFrame
    test_imputed: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    fit_runtime_sec: float = 0.0
    transform_runtime_sec: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolReport:
    """Everything a fold-wise run needs to be auditable."""

    method: str
    protocol: str
    n_folds: int
    folds: List[FoldImputation]
    leak_free: bool = True
    rationale: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "method": self.method,
                    "protocol": self.protocol,
                    "fold_id": f.fold_id,
                    "n_train": len(f.train_idx),
                    "n_test": len(f.test_idx),
                    "fit_runtime_sec": f.fit_runtime_sec,
                    "transform_runtime_sec": f.transform_runtime_sec,
                }
                for f in self.folds
            ]
        )


# --------------------------------------------------------------------------- #
# The protocol itself
# --------------------------------------------------------------------------- #


def impute_within_fold(
    imputer_factory: Callable[[], Any],
    X_missing: pd.DataFrame,
    fold: FoldSpec,
    *,
    protocol: Optional[str] = None,
) -> FoldImputation:
    """Impute one fold under the correct protocol.

    ``imputer_factory`` must return a **fresh, unfitted** imputer on every call.
    That requirement is what makes the fold-independent path genuinely
    independent: the test fold is completed by an object that has never seen a
    training row.

    ``X_missing`` must contain features only -- the outcome column must be
    dropped before this call, otherwise the imputer can reconstruct the label.

    Fold blocks are handed to the imputer with a **reset index**.  Two reasons:
    an imputer must not be able to condition on a row's position in the original
    table (which is precisely the channel reviewer R1-4 objects to in the R0 MAR
    simulator), and several imputers assume a contiguous ``RangeIndex``
    internally.  The original index is restored on the way out.
    """
    train_block = X_missing.iloc[fold.train_idx].copy()
    test_block = X_missing.iloc[fold.test_idx].copy()
    train_index, test_index = train_block.index, test_block.index
    train_block = train_block.reset_index(drop=True)
    test_block = test_block.reset_index(drop=True)

    probe = imputer_factory()
    resolved = protocol or detect_protocol(probe)

    if resolved == PROTOCOL_FIT_TRANSFORM:
        t0 = time.time()
        probe.fit(train_block)
        fit_sec = time.time() - t0

        t1 = time.time()
        train_imp = probe.transform(train_block)
        test_imp = probe.transform(test_block)
        tr_sec = time.time() - t1

        notes = {"fitted_on": "train_fold_only", "n_fit_rows": int(len(train_block))}

    elif resolved == PROTOCOL_FOLD_INDEPENDENT:
        t0 = time.time()
        train_imp = _call_impute(probe, train_block)
        fit_sec = time.time() - t0

        t1 = time.time()
        # A *second*, independently constructed instance: no state, no gradients
        # and no fitted statistics cross the fold boundary.
        test_imputer = imputer_factory()
        if test_imputer is probe:
            raise ValueError(
                "imputer_factory returned the same object twice; the "
                "fold-independent protocol requires a fresh instance per fold "
                "block, otherwise state leaks from train to test."
            )
        test_imp = _call_impute(test_imputer, test_block)
        tr_sec = time.time() - t1

        notes = {
            "fitted_on": "each_block_independently",
            "n_fit_rows": int(len(train_block)),
            "caveat": (
                "transductive method: the test block is completed using the test "
                "block's own observed entries; there is no train->test flow, but "
                "the test block is not scored under an inductive model."
            ),
        }
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown protocol {resolved!r}")

    train_imp = _restore_frame(train_imp, train_index, X_missing.columns)
    test_imp = _restore_frame(test_imp, test_index, X_missing.columns)

    return FoldImputation(
        fold_id=fold.fold_id,
        protocol=resolved,
        train_imputed=train_imp,
        test_imputed=test_imp,
        train_idx=fold.train_idx,
        test_idx=fold.test_idx,
        fit_runtime_sec=float(fit_sec),
        transform_runtime_sec=float(tr_sec),
        notes=notes,
    )


def run_protocol(
    method: str,
    imputer_factory: Callable[[], Any],
    X_missing: pd.DataFrame,
    folds: Iterable[FoldSpec],
    *,
    protocol: Optional[str] = None,
    rationale: str = "",
) -> ProtocolReport:
    """Apply the within-fold protocol across all folds."""
    folds = list(folds)
    results = [
        impute_within_fold(imputer_factory, X_missing, f, protocol=protocol) for f in folds
    ]
    resolved = results[0].protocol if results else (protocol or "unknown")
    return ProtocolReport(
        method=method,
        protocol=resolved,
        n_folds=len(folds),
        folds=results,
        leak_free=True,
        rationale=rationale
        or (
            "fit on train fold, transform test fold"
            if resolved == PROTOCOL_FIT_TRANSFORM
            else "transductive method: train and test folds imputed independently"
        ),
    )


# --------------------------------------------------------------------------- #
# Verification -- the evidence R1-3 asks for
# --------------------------------------------------------------------------- #


def verify_fold_independence(
    imputer_factory: Callable[[], Any],
    X_missing: pd.DataFrame,
    fold: FoldSpec,
    *,
    protocol: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """Empirically prove that the test-fold imputation ignores the train fold.

    The check corrupts the training rows (shuffling them and adding noise to the
    numeric columns), re-runs the protocol, and compares the resulting **test**
    block against the original.  Under a leak-free protocol the two test blocks
    must be identical; under R0's impute-then-split they would differ.

    Returns a dict with ``max_abs_diff`` (numeric columns), ``n_cell_changes``
    (all columns) and ``independent``.  Stochastic imputers must be seeded
    deterministically by the factory for this test to be meaningful; the
    returned ``deterministic_reference`` flag reports whether a repeat run with
    an untouched train block reproduced itself.
    """
    rng = rng or np.random.default_rng(0)

    base = impute_within_fold(imputer_factory, X_missing, fold, protocol=protocol)

    # Determinism reference: same input twice.
    repeat = impute_within_fold(imputer_factory, X_missing, fold, protocol=protocol)
    det = _frames_equal(base.test_imputed, repeat.test_imputed, atol=atol)

    # Corrupt the training rows only.
    perturbed = _corrupt_rows(X_missing, fold.train_idx, rng)

    shifted = impute_within_fold(imputer_factory, perturbed, fold, protocol=protocol)

    max_abs, n_changes = _frame_difference(base.test_imputed, shifted.test_imputed)
    return {
        "protocol": base.protocol,
        "deterministic_reference": bool(det),
        "max_abs_diff": float(max_abs),
        "n_cell_changes": int(n_changes),
        "independent": bool(n_changes == 0),
    }


def _corrupt_rows(X: pd.DataFrame, idx: np.ndarray,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle the given rows among themselves and add large noise to their
    numeric entries; every other row is untouched. The corruption used by both
    directions of the independence check."""
    perturbed = X.copy()
    permuted = perturbed.iloc[idx].sample(
        frac=1.0, random_state=int(rng.integers(1 << 30)))
    permuted.index = perturbed.index[idx]
    perturbed.iloc[idx] = permuted.values
    num_cols = [c for c in perturbed.columns
                if pd.api.types.is_numeric_dtype(perturbed[c])]
    for c in num_cols:
        col = perturbed[c].to_numpy(dtype=float, copy=True)
        scale = np.nanstd(col[idx]) if np.isfinite(np.nanstd(col[idx])) else 1.0
        col[idx] = col[idx] + rng.normal(0.0, max(scale, 1e-6) * 5.0,
                                         size=idx.size)
        perturbed[c] = col
    return perturbed


def verify_independence_per_class(
    imputer_factory: Callable[[], Any],
    X_missing: pd.DataFrame,
    fold: FoldSpec,
    *,
    protocol: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """Per-class independence evidence (second internal review SS9).

    Three executed assertions, with the expected outcome resolved per
    protocol class rather than the blanket ``n_changes == 0`` of
    :func:`verify_fold_independence` (which is correct only for the
    fold-independent class):

    A. **test-feature perturbation -> completed training block unchanged**
       (both classes). This is the direction that certifies "no statistic
       of any test row reaches the completed training table", and with the
       training features and labels unchanged it extends to the downstream
       model fitted on them.
    B. **train-feature perturbation ->** per class:
       fit/transform: inductive test imputations MUST change (positive
       control -- the fitted statistics really do come from the train
       block); fold-independent: test block MUST be bit-identical (the
       independence proof).
    C. Determinism reference: an untouched repeat reproduces itself
       (prerequisite for A and B to be meaningful).

    Labels never enter this code path (``X_missing`` is features only,
    enforced by the callers); the label-permutation assertion is executed
    at the harness level where a label exists.
    """
    rng = rng or np.random.default_rng(0)

    base = impute_within_fold(imputer_factory, X_missing, fold, protocol=protocol)
    repeat = impute_within_fold(imputer_factory, X_missing, fold, protocol=protocol)
    det = (_frames_equal(base.test_imputed, repeat.test_imputed, atol=atol)
           and _frames_equal(base.train_imputed, repeat.train_imputed, atol=atol))

    # A. Corrupt the TEST rows; the completed training block must not move.
    shifted_test = impute_within_fold(
        imputer_factory, _corrupt_rows(X_missing, fold.test_idx, rng),
        fold, protocol=protocol)
    _, n_train_changes = _frame_difference(
        base.train_imputed, shifted_test.train_imputed, atol=atol)

    # B. Corrupt the TRAIN rows; expectation depends on the class.
    shifted_train = impute_within_fold(
        imputer_factory, _corrupt_rows(X_missing, fold.train_idx, rng),
        fold, protocol=protocol)
    _, n_test_changes = _frame_difference(
        base.test_imputed, shifted_train.test_imputed, atol=atol)

    cls = base.protocol
    if cls == PROTOCOL_FOLD_INDEPENDENT:
        b_ok = (n_test_changes == 0)
        b_reading = "test block bit-identical under train corruption (independence proof)"
    else:
        b_ok = (n_test_changes > 0)
        b_reading = ("inductive test imputations change under train corruption "
                     "(positive control: fitted statistics come from the train block)")
    a_ok = (n_train_changes == 0)
    return {
        "protocol": cls,
        "deterministic_reference": bool(det),
        "A_test_perturb_train_block_n_changes": int(n_train_changes),
        "A_pass_train_block_unchanged": bool(a_ok),
        "B_train_perturb_test_block_n_changes": int(n_test_changes),
        "B_pass": bool(b_ok),
        "B_reading": b_reading,
        "independent_per_class": bool(det and a_ok and b_ok),
    }


def _restore_frame(obj, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Put an imputer's output back on the original row index, preserving dtypes."""
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ValueError(f"imputer dropped columns {missing}")
        out = out[list(columns)]
        if len(out) != len(index):
            raise ValueError(f"imputer changed the row count: {len(out)} != {len(index)}")
        out.index = index
        return out
    arr = np.asarray(obj)
    if arr.shape != (len(index), len(columns)):
        raise ValueError(f"imputer returned shape {arr.shape}, expected {(len(index), len(columns))}")
    return pd.DataFrame(arr, index=index, columns=list(columns))


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, *, atol: float = 1e-8) -> bool:
    max_abs, n_changes = _frame_difference(a, b, atol=atol)
    return n_changes == 0


def _frame_difference(
    a: pd.DataFrame, b: pd.DataFrame, *, atol: float = 1e-8
) -> Tuple[float, int]:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return float("inf"), int(max(a.size, b.size))
    max_abs = 0.0
    n_changes = 0
    for c in a.columns:
        sa, sb = a[c], b[c]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            d = np.abs(sa.to_numpy(dtype=float) - sb.to_numpy(dtype=float))
            d = np.where(np.isnan(d), 0.0, d)
            max_abs = max(max_abs, float(d.max()) if d.size else 0.0)
            n_changes += int((d > atol).sum())
        else:
            n_changes += int((sa.astype(str).values != sb.astype(str).values).sum())
    return max_abs, n_changes
