from __future__ import annotations

"""Baseline registry for code_SNI (R1).

Ported from ``project_sni_R0/sni/baselines/registry.py`` with two scientific
changes mandated by P1 T1.4, and nothing else.

1. **Oracle leak removed (P0 finding B6).**
   R0's interface was ``impute(X_complete, X_missing)`` -- the first positional
   argument was the *ground-truth* table (``registry.py:48-52``). Every wrapper
   then either fed it into the implementation or into
   ``fallback_fillna(X_imp, X_complete, ...)``
   (``registry.py:66,82,117,154,196,252,283,331``). SNI was the only method that
   never saw it, so the published comparison handicapped the proposed method.

   The R1 primary interface is::

       impute(X_missing, schema) -> pd.DataFrame

   ``X_complete`` is not a parameter. Everything an imputer needs -- category
   vocabularies, continuous ranges, standardization constants, initial fills --
   comes from :class:`~baselines.schema.ObservedStats`, computed from the
   observed cells of ``X_missing``, plus the role/type declarations in
   ``configs/datasets.yaml``.

2. **fit / transform separation (reviewer R1-3).**
   ``MeanMode``, ``KNN``, ``MICE`` and ``MissForest`` expose
   ``fit(X_train_missing)`` / ``transform(X_new_missing)``. ``GAIN``, ``MIWAE``,
   ``TabCSDI`` and -- after inspecting the package -- ``HyperImpute`` do not, and
   the reason is recorded per method in :data:`FIT_TRANSFORM_ADJUDICATION` rather
   than papered over with a fake ``transform``. Their fallback protocol is
   fold-independent imputation.

The legacy behavior is still reachable via ``legacy_oracle=True`` so that the
before/after impact study (``results/T1.4_deleak``) is a controlled single-factor
contrast rather than a rewrite comparison.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .schema import DataSchema, ObservedStats
from .utils import (
    apply_observed_categories,
    fallback_fillna,
    fallback_fillna_oracle,
    set_categories_from_complete,
)

from .GAIN_v5 import GAINImputer
from .KNN_v1 import knnImputer
from .MeanMode_v1 import MeanModeImputer
from .MICE_v3 import MICEImputer
from .MissForest_v2 import MissForestImputer
from .MIWAE_v3 import MIWAEImputer
from .HyperImpute_v1 import HyperImputeImputer
from .TabCSDI_v1 import TabCSDIImputer

__all__ = [
    "BaseBaseline",
    "build_baseline_imputer",
    "list_baselines",
    "FIT_TRANSFORM_ADJUDICATION",
    "ORACLE_USAGE_R0",
]


# ---------------------------------------------------------------------------
# Documentation tables. These are data, not prose, so that the response letter
# and the P1 report are generated from the same source as the code.
# ---------------------------------------------------------------------------

#: What each method did with ``X_complete`` in R0, and what replaces it in R1.
#: Every ``file:line`` citation in this table and in
#: :data:`FIT_TRANSFORM_ADJUDICATION` refers to the FROZEN R0 tree
#: (``project_sni_R0/sni/baselines/``, tag ``v0.5-R0-submitted``), not to the
#: files in this directory, whose line numbers have moved.
ORACLE_USAGE_R0: Dict[str, Dict[str, str]] = {
    "MeanMode": {
        "r0_use": "Imputation values themselves: column mean and mode computed "
                  "directly on the ground-truth table.",
        "r0_evidence": "registry.py:64-67 -> MeanMode_v1.py:59-60 (mean), :68-69 (mode)",
        "severity": "direct",
        "r1_replacement": "ObservedStats.cont_mean / cat_mode from the observed "
                          "cells of X_missing (MeanMode_v1.fit/transform).",
    },
    "KNN": {
        "r0_use": "Continuous ranges (max-min) used to normalize the Gower "
                  "distance, computed on the ground-truth table.",
        "r0_evidence": "registry.py:80-83 -> KNN_v1.py:40-48 called at :110",
        "severity": "indirect",
        "r1_replacement": "ObservedStats.cont_range (observed min/max) installed "
                          "via knnImputer.set_ranges_from_stats.",
    },
    "MICE": {
        "r0_use": "LabelEncoder vocabulary AND the initial fill values "
                  "(mean/mode) of the chained-equations loop.",
        "r0_evidence": "registry.py:114-118 -> MICE_v3.py:226-228 (vocabulary), "
                       ":250 (mean), :254 (mode)",
        "severity": "direct",
        "r1_replacement": "ObservedStats.categories / cont_mean / cat_mode; "
                          "MICE_v3.impute(..., stats=...).",
    },
    "MissForest": {
        "r0_use": "Categorical vocabulary only. The implementation's own comment "
                  "(MissForest_v2.py:392-393) says X_complete is unused, and the "
                  "body indeed never reads it -- but registry.py:152 stamped the "
                  "oracle vocabulary onto the categorical dtype of the frame it "
                  "was given, which MissForest_v2.py:177 then adopts.",
        "r0_evidence": "registry.py:150-155 -> utils.set_categories_from_complete "
                       "-> MissForest_v2.py:176-177",
        "severity": "vocabulary",
        "r1_replacement": "apply_observed_categories(X_missing, stats) before the "
                          "call; initial fill was already observed-only (:185, :193).",
    },
    "GAIN": {
        "r0_use": "Min-max normalization constants fitted on the ground-truth "
                  "matrix, plus the one-hot vocabulary. R0's docstring called "
                  "X_complete a pass-through; it is not.",
        "r0_evidence": "registry.py:193-198 -> GAIN_v5.py:192-251 (_prepare_data "
                       "builds the truth matrix, vocabulary at :213-215) and "
                       ":374 (_normalize_data(data_complete, fit=True))",
        "severity": "direct",
        "r1_replacement": "GAIN_v5.impute(None, X_missing): the normaliser is "
                          "fitted on the incomplete matrix (nanmin/nanmax at "
                          ":93-94 already ignore missing cells) and the "
                          "vocabulary comes from ObservedStats.",
    },
    "MIWAE": {
        "r0_use": "Categorical vocabulary only, via the dtype installed by "
                  "set_categories_from_complete. The body reads only "
                  "X_incomplete.shape.",
        "r0_evidence": "registry.py:249-254 -> MIWAE_v3.py:379-380 (shape check) "
                       "and _prepare_data's df[col].astype('category')",
        "severity": "vocabulary",
        "r1_replacement": "apply_observed_categories(...) + impute(None, X_missing); "
                          "continuous standardization was already observed-only.",
    },
    "HyperImpute": {
        "r0_use": "Categorical mapping (full label set).",
        "r0_evidence": "registry.py:280-284 -> HyperImpute_v1.py:90, :113-115 "
                       "(all line numbers refer to project_sni_R0/sni/baselines/)",
        "severity": "vocabulary",
        "r1_replacement": "Observed vocabulary from ObservedStats; "
                          "HyperImpute_v1.impute(None, X_missing).",
    },
    "TabCSDI": {
        "r0_use": "Per-column mean and std used to standardise the diffusion "
                  "input, plus the categorical label set.",
        "r0_evidence": "registry.py:328-332 -> TabCSDI_v1.py:310-312 (mean/std), "
                       ":329-330 (categories)",
        "severity": "direct",
        "r1_replacement": "Observed mean/std and observed vocabulary; "
                          "TabCSDI_v1.impute(None, X_missing).",
    },
    "_ALL": {
        "r0_use": "Every wrapper finished with fallback_fillna(X_imp, X_complete, "
                  "...), filling residual NaNs from the ground truth.",
        "r0_evidence": "registry.py:66,82,117,154,196,252,283,331 -> "
                       "baselines/utils.py:53-87",
        "severity": "direct",
        "r1_replacement": "baselines/utils.py:fallback_fillna(X_imp, stats, ...) "
                          "using the observed mean/mode.",
    },
}


#: Per-method adjudication for reviewer R1-3. Written to be quotable verbatim.
FIT_TRANSFORM_ADJUDICATION: Dict[str, Dict[str, Any]] = {
    "MeanMode": {
        "separable": True,
        "mechanism": "closed-form column statistic",
        "fitted": "per-column observed mean (continuous) and mode (categorical)",
        "transformed": "missing cells of the new frame are filled with the stored "
                       "constants; nothing is recomputed",
        "reason": None,
    },
    "KNN": {
        "separable": True,
        "mechanism": "instance-based; donor pool + distance normalization",
        "fitted": "continuous ranges for the Gower denominator, and the imputed "
                  "training fold retained as the donor pool",
        "transformed": "each new row is scored against the fixed donor pool; new "
                       "rows are never donors for one another",
        "reason": None,
    },
    "MICE": {
        "separable": True,
        "mechanism": "chained equations; imputation sequence replay",
        "fitted": "label encoders, initial fill values, and for every "
                  "(iteration, column) either the posterior mean beta_hat with "
                  "its Cholesky factor, residual scale, training predicted means "
                  "and donor values (PMM branch) or the fitted logistic model",
        "transformed": "the stored sequence is replayed in order on the new "
                       "frame; no regression is refitted",
        "reason": None,
    },
    "MissForest": {
        "separable": True,
        "mechanism": "iterative random forests; imputation sequence replay",
        "fitted": "category codes, initial fill values, and the random forest "
                  "fitted at each (iteration, column) up to the gamma stopping "
                  "point",
        "transformed": "the stored forests predict on the new frame in the same "
                       "order; no forest is refitted",
        "reason": None,
    },
    "HyperImpute": {
        "separable": False,
        "mechanism": "AutoML column-model search; package exposes a fit/transform "
                     "facade that does not separate",
        "fitted": None,
        "transformed": None,
        "reason": (
            "Separable in principle -- HyperImpute is an iterative per-column "
            "regression, structurally the same shape as MICE and MissForest, both "
            "of which we did separate. It is not separable in the reference "
            "package. hyperimpute's HyperImputePlugin advertises the sklearn "
            "estimator API (it inherits sklearn.impute._base._BaseImputer), but "
            "plugin_hyperimpute.py:124-125 defines _fit as `return self` -- it "
            "stores nothing -- and :127-128 defines _transform as "
            "`self.model.fit_transform(X)`, which reruns the whole AutoML search "
            "on the frame passed to transform. We verified this empirically: "
            "after p.fit(train), p.transform(test) is elementwise identical to a "
            "fresh plugin's fit_transform(test), so the training fold has no "
            "influence at all. Implementing genuine separation would mean "
            "re-implementing the column-model search, which is HyperImpute's "
            "entire contribution; we would then no longer be benchmarking "
            "HyperImpute. This is a defect of the published package, not of our "
            "harness, and we report it as such."
        ),
    },
    "GAIN": {
        "separable": False,
        "mechanism": "adversarial single-table training",
        "fitted": None,
        "transformed": None,
        "reason": (
            "Three blockers, in decreasing order of how fundamental they are. "
            "(i) Objective: the discriminator is trained to recover the mask of "
            "the table it is fitted on, and the hint mechanism "
            "(GAIN_v5.py:262-272) samples that same mask; the generator's "
            "adversarial term is therefore defined relative to one table's "
            "missingness pattern, so applying it to a fold with a different "
            "pattern is out-of-distribution by construction. "
            "(ii) State: the generator and discriminator are constructed inside "
            "impute() (GAIN_v5.py:395-396) and the min-max normaliser is fitted "
            "there too (:374); nothing survives the call, and there is no "
            "inference entry point. "
            "(iii) Encoding: the one-hot width is derived from the vocabulary of "
            "the frame being imputed (:212-216), so a level present only in the "
            "test fold has no slot. "
            "Blockers (ii) and (iii) are engineering; (i) is methodological. We "
            "do not add a transform path because doing so would replace the "
            "published GAIN protocol with one of our own invention, which is "
            "exactly the criticism we are trying to answer."
        ),
    },
    "MIWAE": {
        "separable": False,
        "mechanism": "importance-weighted VAE, single-table training",
        "fitted": None,
        "transformed": None,
        "reason": (
            "MIWAE's encoder/decoder are amortised, so a transform path is "
            "conceivable, but it does not exist in the reference implementation "
            "and cannot be added without redefining the method: the encoder and "
            "decoder are constructed inside impute() (MIWAE_v3.py:397-409), the "
            "continuous normaliser is fitted inside _prepare_data, the one-hot "
            "vocabulary is read off the frame under imputation, and the "
            "imputation step is a self-normalized importance-weighted average "
            "over L=10000 draws taken during the same call. Carrying state would "
            "mean persisting the encoder, the decoder, the normaliser and the "
            "vocabulary, and writing a new inference routine -- i.e. shipping a "
            "variant of MIWAE rather than MIWAE."
        ),
    },
    "TabCSDI": {
        "separable": False,
        "mechanism": "conditional score-based diffusion, single-table training",
        "fitted": None,
        "transformed": None,
        "reason": (
            "The denoiser is conditioned on the observed entries of the table it "
            "was trained on: _encode(..., fit=True) is called from impute() "
            "(TabCSDI_v1.py:597) and fixes the standardization constants, the "
            "label set and the encoded width in the same call that trains the "
            "model and runs the reverse diffusion. No fitted object outlives "
            "impute(), and the reverse process is executed over the training "
            "matrix's mask. As with MIWAE, a transform path would be new code "
            "rather than the published method."
        ),
    },
}

#: Protocol used for the three non-separable methods.
FOLD_INDEPENDENT_PROTOCOL = (
    "Fold-independent imputation. The training fold and the test fold are each "
    "imputed by a separate instance of the method, fitted only on its own fold. "
    "No parameter, statistic, donor or gradient crosses the fold boundary, so the "
    "test fold's imputed values are a function of the test fold's observed cells "
    "alone. This is weaker than fit/transform -- it does not reproduce the "
    "deployment situation in which a fixed imputer meets one new record -- and we "
    "say so explicitly rather than presenting it as equivalent."
)


def _filter_kwargs(kwargs: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
    """Return a new kwargs dict containing only the allowed keys with non-NaN values."""
    out: Dict[str, Any] = {}
    for k in allowed:
        if k not in kwargs:
            continue
        v = kwargs[k]
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class BaseBaseline:
    """Common interface.

    Primary (R1)::

        imp = build_baseline_imputer("MICE", categorical_vars=..., continuous_vars=...)
        X_imputed = imp.impute(X_missing, schema)

    Fit/transform (R1-3), where supported::

        imp.fit(X_train_missing, schema)
        X_test_imputed = imp.transform(X_test_missing)

    Legacy (R0 reproduction only)::

        imp = build_baseline_imputer(..., legacy_oracle=True)
        X_imputed = imp.impute_legacy(X_complete, X_missing)
    """

    categorical_vars: List[str]
    continuous_vars: List[str]
    legacy_oracle: bool = False

    method: str = field(init=False, default="")
    supports_fit_transform: bool = field(init=False, default=False)

    # -- helpers -------------------------------------------------------
    def _resolve_schema(self, schema: Optional[DataSchema]) -> DataSchema:
        if schema is not None:
            return schema
        return DataSchema.from_var_lists(self.categorical_vars, self.continuous_vars)

    def _fit_stats(self, X_missing: pd.DataFrame, schema: Optional[DataSchema]) -> ObservedStats:
        sch = self._resolve_schema(schema)
        stats = ObservedStats.from_frame(X_missing, sch)
        self.schema_ = sch
        self.stats_ = stats
        return stats

    # -- API -----------------------------------------------------------
    def impute(self, X_missing: pd.DataFrame, schema: Optional[DataSchema] = None) -> pd.DataFrame:
        """De-leaked imputation. Never receives the ground-truth table."""
        raise NotImplementedError

    def impute_legacy(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        """R0 behavior, verbatim. Only for the before/after impact study."""
        raise NotImplementedError

    def fit(self, X_train_missing: pd.DataFrame, schema: Optional[DataSchema] = None) -> "BaseBaseline":
        raise NotImplementedError(
            f"{self.method} does not support fit/transform. "
            f"Reason: {FIT_TRANSFORM_ADJUDICATION[self.method]['reason']}"
        )

    def transform(self, X_new_missing: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            f"{self.method} does not support fit/transform. "
            f"Use the fold-independent protocol: {FOLD_INDEPENDENT_PROTOCOL}"
        )

    # -- dispatch used by the runner ----------------------------------
    def run(self, X_missing: pd.DataFrame, schema=None, X_complete=None) -> pd.DataFrame:
        if self.legacy_oracle:
            if X_complete is None:
                raise ValueError("legacy_oracle=True requires X_complete")
            return self.impute_legacy(X_complete, X_missing)
        return self.impute(X_missing, schema)


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------

@dataclass
class MeanModeBaseline(BaseBaseline):
    def __post_init__(self):
        self.method = "MeanMode"
        self.supports_fit_transform = True
        self._impl = MeanModeImputer(self.categorical_vars, self.continuous_vars)

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        self._impl.fit(X_missing, stats=stats)
        X_imp, _ = self._impl.transform(X_missing)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def fit(self, X_train_missing, schema=None):
        stats = self._fit_stats(X_train_missing, schema)
        self._impl.fit(X_train_missing, stats=stats)
        return self

    def transform(self, X_new_missing):
        X_imp, _ = self._impl.transform(X_new_missing)
        return fallback_fillna(X_imp, self.stats_, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        X_imp, _ = self._impl.impute(X_complete, X_missing)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class KNNBaseline(BaseBaseline):
    k: int = 5

    def __post_init__(self):
        self.method = "KNN"
        self.supports_fit_transform = True
        self._impl = knnImputer(self.categorical_vars, self.continuous_vars, k=int(self.k))

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        X_imp, _ = self._impl.impute_deleaked(X_missing, stats)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def fit(self, X_train_missing, schema=None):
        stats = self._fit_stats(X_train_missing, schema)
        self._impl.fit(X_train_missing, stats)
        return self

    def transform(self, X_new_missing):
        X_imp, _ = self._impl.transform(X_new_missing)
        return fallback_fillna(X_imp, self.stats_, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        X_imp, _ = self._impl.impute(X_complete, X_missing)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class MICEBaseline(BaseBaseline):
    """MICE v3 with predictive mean matching. Van Buuren & Groothuis-Oudshoorn (2011)."""

    max_iter: int = 5
    seed: int = 42
    donors: int = 5
    matchtype: int = 1
    ridge: float = 1e-5

    def __post_init__(self):
        self.method = "MICE"
        self.supports_fit_transform = True
        self._impl = MICEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            max_iter=int(self.max_iter),
            seed=int(self.seed),
            donors=int(self.donors),
            matchtype=int(self.matchtype),
            ridge=float(self.ridge),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        X_imp, _ = self._impl.impute(None, X_missing, stats=stats)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def fit(self, X_train_missing, schema=None):
        stats = self._fit_stats(X_train_missing, schema)
        self._impl.fit(X_train_missing, stats)
        return self

    def transform(self, X_new_missing):
        X_imp, _ = self._impl.transform(X_new_missing)
        return fallback_fillna(X_imp, self.stats_, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class MissForestBaseline(BaseBaseline):
    """MissForest v2 with the gamma stopping criterion. Stekhoven & Buhlmann (2012)."""

    n_estimators: int = 100
    max_iter: int = 10
    seed: int = 42
    n_jobs: int = -1
    verbose: bool = False
    decreasing: bool = False

    def __post_init__(self):
        self.method = "MissForest"
        self.supports_fit_transform = True
        self._impl = MissForestImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            n_estimators=int(self.n_estimators),
            max_iter=int(self.max_iter),
            seed=int(self.seed),
            n_jobs=int(self.n_jobs),
            verbose=bool(self.verbose),
            decreasing=bool(self.decreasing),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        Xm = apply_observed_categories(X_missing, stats)
        X_imp, _ = self._impl.impute(Xm)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def fit(self, X_train_missing, schema=None):
        stats = self._fit_stats(X_train_missing, schema)
        self._impl.fit(apply_observed_categories(X_train_missing, stats), stats)
        return self

    def transform(self, X_new_missing):
        Xm = apply_observed_categories(X_new_missing, self.stats_)
        X_imp, _ = self._impl.transform(Xm)
        return fallback_fillna(X_imp, self.stats_, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        _, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class GAINBaseline(BaseBaseline):
    """GAIN with the official hyperparameters. Yoon et al. (2018), ICML."""

    seed: int = 42
    use_gpu: bool = False
    hidden_dim: int = 256
    batch_size: int = 128
    hint_rate: float = 0.9
    alpha: float = 100.0
    iterations: int = 10000
    learning_rate: float = 1e-3

    def __post_init__(self):
        self.method = "GAIN"
        self.supports_fit_transform = False
        self._impl = GAINImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dim=int(self.hidden_dim),
            batch_size=int(self.batch_size),
            hint_rate=float(self.hint_rate),
            alpha=float(self.alpha),
            iterations=int(self.iterations),
            learning_rate=float(self.learning_rate),
            seed=int(self.seed),
            use_gpu=bool(self.use_gpu),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        Xm = apply_observed_categories(X_missing, stats)
        X_imp, _ = self._impl.impute(None, Xm)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class MIWAEBaseline(BaseBaseline):
    """MIWAE v3. Mattei & Frellsen (2019), ICML."""

    seed: int = 42
    use_gpu: bool = False
    hidden_dims: str = "128,128,128"
    latent_dim: int = 10
    num_iw_samples: int = 20
    num_impute_samples: int = 10000
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 500
    min_epochs: int = 200
    min_variance: float = 0.01

    def __post_init__(self):
        self.method = "MIWAE"
        self.supports_fit_transform = False
        if isinstance(self.hidden_dims, str):
            hidden_dims_list = [int(x.strip()) for x in self.hidden_dims.split(",")]
        else:
            hidden_dims_list = list(self.hidden_dims)

        self._impl = MIWAEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dims=hidden_dims_list,
            latent_dim=int(self.latent_dim),
            num_iw_samples=int(self.num_iw_samples),
            num_impute_samples=int(self.num_impute_samples),
            lr=float(self.lr),
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            min_epochs=int(self.min_epochs),
            seed=int(self.seed),
            use_gpu=bool(self.use_gpu),
            min_variance=float(self.min_variance),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        Xm = apply_observed_categories(X_missing, stats)
        X_imp, _ = self._impl.impute(None, Xm)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class HyperImputeBaseline(BaseBaseline):
    """HyperImpute (AutoML iterative imputation). Jarrett et al. (2022), ICML."""

    seed: int = 42
    timeout: int = 600
    optimizer: str = "hyperband"

    def __post_init__(self):
        self.method = "HyperImpute"
        # See FIT_TRANSFORM_ADJUDICATION["HyperImpute"]: the package's fit() is a
        # no-op and its transform() refits from scratch, so exposing fit/transform
        # here would advertise a separation that does not exist.
        self.supports_fit_transform = False
        self._impl = HyperImputeImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            seed=int(self.seed),
            timeout=int(self.timeout),
            optimizer=str(self.optimizer),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        Xm = apply_observed_categories(X_missing, stats)
        X_imp, _ = self._impl.impute(None, Xm)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def fit_package_facade(self, X_train_missing, schema=None):
        """Evidence-only. NOT a real fit -- see the adjudication entry."""
        stats = self._fit_stats(X_train_missing, schema)
        self._impl.fit(apply_observed_categories(X_train_missing, stats), stats)
        return self

    def transform_package_facade(self, X_new_missing):
        """Evidence-only. Refits the AutoML search on ``X_new_missing``."""
        Xm = apply_observed_categories(X_new_missing, self.stats_)
        X_imp, _ = self._impl.transform(Xm)
        return fallback_fillna(X_imp, self.stats_, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


@dataclass
class TabCSDIBaseline(BaseBaseline):
    """TabCSDI. Tashiro et al. (2021) CSDI; Zheng & Charoenphakdee (2022) TabCSDI."""

    seed: int = 42
    use_gpu: bool = False
    diffusion_steps: int = 50
    n_samples: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3

    def __post_init__(self):
        self.method = "TabCSDI"
        self.supports_fit_transform = False
        self._impl = TabCSDIImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            seed=int(self.seed),
            use_gpu=bool(self.use_gpu),
            diffusion_steps=int(self.diffusion_steps),
            n_samples=int(self.n_samples),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            lr=float(self.lr),
        )

    def impute(self, X_missing, schema=None):
        stats = self._fit_stats(X_missing, schema)
        Xm = apply_observed_categories(X_missing, stats)
        X_imp, _ = self._impl.impute(None, Xm)
        return fallback_fillna(X_imp, stats, self.categorical_vars, self.continuous_vars)

    def impute_legacy(self, X_complete, X_missing):
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        return fallback_fillna_oracle(X_imp, X_complete, self.categorical_vars, self.continuous_vars)


_REGISTRY = {
    "MeanMode": MeanModeBaseline,
    "KNN": KNNBaseline,
    "MICE": MICEBaseline,
    "MissForest": MissForestBaseline,
    "GAIN": GAINBaseline,
    "MIWAE": MIWAEBaseline,
    "HyperImpute": HyperImputeBaseline,
    "TabCSDI": TabCSDIBaseline,
}


def list_baselines() -> List[str]:
    return sorted(_REGISTRY.keys())


def build_baseline_imputer(
    method: str,
    categorical_vars: Sequence[str],
    continuous_vars: Sequence[str],
    *,
    seed: int = 42,
    use_gpu: bool = False,
    legacy_oracle: bool = False,
    **kwargs: Any,
) -> BaseBaseline:
    """Build a baseline imputer by name.

    ``legacy_oracle=True`` selects the R0 behavior (``impute_legacy``); it exists
    only so the impact study can measure the leak. All production paths leave it
    at ``False``.
    """
    m = str(method).strip()
    if m not in _REGISTRY:
        raise KeyError(f"Unknown baseline method '{method}'. Available: {list_baselines()}")

    cls = _REGISTRY[m]

    if m == "KNN":
        allowed = ["k"]
    elif m == "MICE":
        allowed = ["max_iter", "donors", "matchtype", "ridge"]
    elif m == "MissForest":
        allowed = ["n_estimators", "max_iter", "n_jobs", "verbose", "decreasing"]
    elif m == "GAIN":
        allowed = ["hidden_dim", "batch_size", "hint_rate", "alpha", "iterations", "learning_rate"]
    elif m == "MIWAE":
        allowed = ["hidden_dims", "latent_dim", "num_iw_samples", "num_impute_samples",
                   "lr", "batch_size", "epochs", "min_epochs", "min_variance"]
    elif m == "HyperImpute":
        allowed = ["timeout", "optimizer"]
    elif m == "TabCSDI":
        allowed = ["diffusion_steps", "n_samples", "d_model", "n_heads", "n_layers",
                   "epochs", "batch_size", "lr"]
    else:
        allowed = []

    filtered = _filter_kwargs(kwargs, allowed)

    # Baselines that accept seed/use_gpu (unchanged from R0; note that MeanMode
    # and KNN take no seed at all, which is why they are deterministic given the
    # mask -- see results/T1.4_deleak).
    if m in {"MICE", "MissForest", "GAIN", "MIWAE", "HyperImpute", "TabCSDI"}:
        filtered.update({"seed": int(seed)})
    if m in {"GAIN", "MIWAE", "TabCSDI"}:
        filtered.update({"use_gpu": bool(use_gpu)})

    return cls(
        categorical_vars=list(categorical_vars),
        continuous_vars=list(continuous_vars),
        legacy_oracle=bool(legacy_oracle),
        **filtered,
    )
