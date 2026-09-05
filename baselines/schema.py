from __future__ import annotations

"""Schema and observed-statistics layer for the R1 baselines.

Purpose (P1 T1.4 step 2, P0 finding B6)
---------------------------------------
In R0 every baseline received the *ground-truth* table as the first positional
argument of ``impute`` (``project_sni_R0/sni/baselines/registry.py:48-52``) and
several of them read real statistics out of it: means, modes, min/max ranges,
standardization constants and the full category vocabulary. SNI was the only
method that never touched it, so the published comparison was biased *against*
the proposed method.

This module defines the two objects that replace ``X_complete``:

``DataSchema``
    Column *roles and types only* -- which columns are continuous, which are
    categorical, which are identifiers. Read from ``configs/datasets.yaml``
    (engineering principle E1) or built directly from the manifest's
    ``categorical_vars`` / ``continuous_vars`` lists.

``ObservedStats``
    Every *numeric* quantity a de-leaked imputer is allowed to use: category
    vocabularies, means, standard deviations, min/max, ranges and modes. All of
    them are computed from the **observed cells of the incomplete table only**.

Deliberate design decision -- why the schema carries no numbers
--------------------------------------------------------------
``configs/datasets.yaml`` also records, per column, a ``range`` and a
``cardinality``. Those were derived from the complete tables, so feeding them
into an imputer would re-open exactly the leak we are closing, only laundered
through a YAML file. They are therefore treated here as **documentation and
validation metadata, never as imputer input**. ``DataSchema`` exposes them under
``declared_range`` / ``declared_cardinality`` purely so that audit code
(``results/T1.4_deleak``) can *measure* the gap between the oracle statistic and
the observed one; :class:`ObservedStats` never reads them.

The one exception the design allows for is ``declared_categories``: a genuine
external codebook (e.g. "NHANES gender_std is coded {0, 1} by definition") is
domain knowledge available at deployment time and is not a leak. No dataset in
``datasets.yaml`` currently declares one, so in R1 this hook is unused and every
vocabulary comes from the observed data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["DataSchema", "ObservedStats"]


def _sorted_unique(values: Iterable[Any]) -> List[Any]:
    """Deterministic vocabulary ordering, mirroring R0's behavior.

    R0 sorted category sets where possible (``baselines/utils.py:41-45``); we keep
    the same convention so that integer-coded categories map to the same codes as
    before and the before/after comparison isolates the leak rather than a
    relabelling artifact.
    """
    vals = pd.Series(list(values)).dropna().unique().tolist()
    try:
        vals = sorted(vals)
    except Exception:
        pass
    return vals


@dataclass
class DataSchema:
    """Column roles and types. Carries no data-derived numbers."""

    continuous_vars: List[str]
    categorical_vars: List[str]
    identifier_column: Optional[str] = None
    dataset: Optional[str] = None
    #: Optional external codebook: {column: [category, ...]}. Domain knowledge,
    #: not data-derived. Empty in R1.
    declared_categories: Dict[str, List[Any]] = field(default_factory=dict)
    #: Audit metadata copied verbatim from datasets.yaml. NEVER used for imputation.
    declared_range: Dict[str, Sequence[float]] = field(default_factory=dict)
    declared_cardinality: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_var_lists(
        cls,
        categorical_vars: Sequence[str],
        continuous_vars: Sequence[str],
        *,
        dataset: Optional[str] = None,
    ) -> "DataSchema":
        """Build from the manifest-style variable lists used by R0."""
        return cls(
            continuous_vars=[str(c).strip() for c in continuous_vars if str(c).strip()],
            categorical_vars=[str(c).strip() for c in categorical_vars if str(c).strip()],
            dataset=dataset,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, dataset: str) -> "DataSchema":
        """Build from ``configs/datasets.yaml`` (E1: configs are the single source of truth)."""
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        try:
            spec = cfg["datasets"][dataset]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise KeyError(
                f"dataset {dataset!r} not declared in {path}; available: "
                f"{sorted(cfg.get('datasets', {}))}"
            ) from exc

        cont: List[str] = []
        cat: List[str] = []
        declared_range: Dict[str, Sequence[float]] = {}
        declared_card: Dict[str, int] = {}
        declared_cats: Dict[str, List[Any]] = {}

        for col, meta in (spec.get("columns") or {}).items():
            role = str(meta.get("role", "imputable"))
            ctype = str(meta.get("type", ""))
            if role != "imputable":
                continue
            if ctype == "continuous":
                cont.append(col)
            elif ctype == "categorical":
                cat.append(col)
            else:  # integer_index or anything else -> not imputable material
                continue
            if meta.get("range") is not None:
                declared_range[col] = list(meta["range"])
            if meta.get("cardinality") is not None:
                declared_card[col] = int(meta["cardinality"])
            if meta.get("categories") is not None:
                declared_cats[col] = list(meta["categories"])

        return cls(
            continuous_vars=cont,
            categorical_vars=cat,
            identifier_column=spec.get("identifier_column"),
            dataset=dataset,
            declared_categories=declared_cats,
            declared_range=declared_range,
            declared_cardinality=declared_card,
        )

    # ------------------------------------------------------------------
    @property
    def all_vars(self) -> List[str]:
        return list(self.continuous_vars) + list(self.categorical_vars)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.all_vars if c not in df.columns]
        if missing:
            raise KeyError(f"columns declared in schema but absent from frame: {missing}")


@dataclass
class ObservedStats:
    """Sufficient statistics derived from the observed cells of an incomplete table.

    This is the complete set of information a de-leaked baseline is allowed to
    carry from data. It is also the object that makes ``fit`` / ``transform``
    separation meaningful: ``fit`` produces one of these on the training fold and
    ``transform`` reuses it verbatim on the test fold.
    """

    categories: Dict[str, List[Any]]
    cat_mode: Dict[str, Any]
    cont_mean: Dict[str, float]
    cont_std: Dict[str, float]
    cont_min: Dict[str, float]
    cont_max: Dict[str, float]
    cont_range: Dict[str, float]
    n_rows: int
    n_observed: Dict[str, int]
    #: "observed" for the de-leaked path, "oracle" when built from ground truth
    #: (only ever used by the legacy comparison path).
    source: str = "observed"

    @classmethod
    def from_frame(
        cls,
        X: pd.DataFrame,
        schema: DataSchema,
        *,
        source: str = "observed",
    ) -> "ObservedStats":
        """Compute statistics from ``X`` ignoring NaN cells.

        When ``X`` is the incomplete table this yields the de-leaked statistics.
        The legacy comparison path passes the ground-truth table with
        ``source="oracle"``; nothing else about the computation differs, which is
        what makes the before/after comparison a clean single-factor contrast.
        """
        schema.validate(X)

        categories: Dict[str, List[Any]] = {}
        cat_mode: Dict[str, Any] = {}
        cont_mean: Dict[str, float] = {}
        cont_std: Dict[str, float] = {}
        cont_min: Dict[str, float] = {}
        cont_max: Dict[str, float] = {}
        cont_range: Dict[str, float] = {}
        n_observed: Dict[str, int] = {}

        for col in schema.continuous_vars:
            vals = pd.to_numeric(X[col], errors="coerce")
            arr = vals.dropna().to_numpy(dtype=float)
            n_observed[col] = int(arr.size)
            if arr.size == 0:
                cont_mean[col] = 0.0
                cont_std[col] = 1.0
                cont_min[col] = 0.0
                cont_max[col] = 0.0
                cont_range[col] = 1.0
                continue
            cont_mean[col] = float(np.mean(arr))
            std = float(np.std(arr))
            cont_std[col] = std if std > 1e-8 else 1.0
            cont_min[col] = float(np.min(arr))
            cont_max[col] = float(np.max(arr))
            rng = cont_max[col] - cont_min[col]
            cont_range[col] = rng if rng > 0 else 1.0

        for col in schema.categorical_vars:
            if col in schema.declared_categories:
                # External codebook (domain knowledge). Unused in R1.
                cats = list(schema.declared_categories[col])
            else:
                cats = _sorted_unique(X[col])
            categories[col] = cats
            obs = X[col].dropna()
            n_observed[col] = int(obs.shape[0])
            if obs.shape[0] == 0:
                cat_mode[col] = cats[0] if cats else np.nan
            else:
                vc = obs.value_counts(dropna=True)
                # value_counts is already sorted by count desc; ties broken by the
                # first occurrence, matching pandas ``Series.mode().iloc[0]``
                # closely enough that R0 and R1 agree on every column tested.
                mode_series = obs.mode(dropna=True)
                cat_mode[col] = mode_series.iloc[0] if len(mode_series) else vc.index[0]

        return cls(
            categories=categories,
            cat_mode=cat_mode,
            cont_mean=cont_mean,
            cont_std=cont_std,
            cont_min=cont_min,
            cont_max=cont_max,
            cont_range=cont_range,
            n_rows=int(X.shape[0]),
            n_observed=n_observed,
            source=source,
        )

    # ------------------------------------------------------------------
    def apply_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` whose categorical columns use this vocabulary.

        R1 replacement for ``baselines/utils.py:set_categories_from_complete``.
        Categories present in ``X`` but absent from the fitted vocabulary become
        NaN, which is the correct and honest behavior under fit/transform: a
        model fitted on the training fold has no parameter for a label it never
        saw. The count of such cells is reported by :meth:`unseen_category_count`.
        """
        out = X.copy()
        for col, cats in self.categories.items():
            if col not in out.columns:
                continue
            out[col] = pd.Categorical(out[col], categories=cats)
        return out

    def unseen_category_count(self, X: pd.DataFrame) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for col, cats in self.categories.items():
            if col not in X.columns:
                continue
            known = set(cats)
            obs = X[col].dropna()
            counts[col] = int(sum(1 for v in obs if v not in known))
        return counts

    def to_dict(self) -> Dict[str, Any]:
        def _clean(d: Mapping[str, Any]) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                out[k] = v
            return out

        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "categories": {k: [int(x) if isinstance(x, np.integer) else x for x in v]
                           for k, v in self.categories.items()},
            "cat_mode": _clean(self.cat_mode),
            "cont_mean": _clean(self.cont_mean),
            "cont_std": _clean(self.cont_std),
            "cont_min": _clean(self.cont_min),
            "cont_max": _clean(self.cont_max),
            "cont_range": _clean(self.cont_range),
            "n_observed": _clean(self.n_observed),
        }
