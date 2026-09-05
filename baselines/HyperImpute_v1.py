# HyperImpute_v1.py
# -*- coding: utf-8 -*-
"""
HyperImpute - Automated imputation via AutoML column-wise model selection.

Wrapper around the ``hyperimpute`` package that provides a consistent
interface matching the other baselines in this repository.

HyperImpute automatically selects the best imputation model per column
via Hyperband-based search over a configurable set of learners.

Reference:
    Jarrett, D., Cebere, B. C., Liu, T., Curth, A., & van der Schaar, M. (2022).
    HyperImpute: Generalized Iterative Imputation with Automatic Model Selection.
    ICML 2022.

Package: pip install hyperimpute
"""

from __future__ import annotations

import logging
import signal
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class HyperImputeImputer:
    """
    HyperImpute imputer using the official ``hyperimpute`` package.

    Key characteristics:
    - AutoML-based: selects the best model per column via Hyperband
    - Does not distinguish continuous/categorical internally (auto-detected)
    - We encode categoricals to integer codes before calling the plugin
      and decode back afterward (consistent with MICE/MissForest)
    """

    def __init__(
        self,
        categorical_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        seed: int = 42,
        timeout: int = 600,
        optimizer: str = "hyperband",
        classifier_seed: Optional[List[str]] = None,
        regression_seed: Optional[List[str]] = None,
    ):
        """
        Args:
            categorical_vars: Column names of categorical features.
            continuous_vars: Column names of continuous features.
            seed: Random seed for reproducibility.
            timeout: Maximum runtime in seconds (default 600).
            optimizer: Search strategy ('hyperband' or 'simple').
            classifier_seed: Classifier candidates for categorical columns.
                Defaults to None → let hyperimpute use its built-in defaults.
                Valid names (hyperimpute >=0.1): "logistic_regression",
                "random_forest", "xgboost", "catboost".
            regression_seed: Regressor candidates for continuous columns.
                Defaults to None → let hyperimpute use its built-in defaults.
                Valid names (hyperimpute >=0.1): "linear_regression",
                "random_forest_regressor", "xgboost_regressor",
                "catboost_regressor".
        """
        self.categorical_vars = list(categorical_vars or [])
        self.continuous_vars = list(continuous_vars or [])
        self.seed = seed
        self.timeout = timeout
        self.optimizer = optimizer
        # None → hyperimpute uses its own internal defaults (safest option).
        self.classifier_seed = classifier_seed
        self.regression_seed = regression_seed
        self.models_: Dict = {}

    def impute(
        self,
        X_complete,
        X_missing: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run HyperImpute on *X_missing*.

        Args:
            X_complete: Ground-truth DataFrame, or ``None`` for the R1 de-leaked
                path. R0 read the full categorical label set off it at
                ``:114-115`` -- "used only for category mapping" is accurate but
                it is still the oracle: under a mask that hides every instance of
                a rare level, R0's HyperImpute still had a code for it. With
                ``None`` the mapping comes from the observed cells of
                ``X_missing``.
            X_missing: DataFrame with NaN at missing positions.

        Returns:
            X_imputed: Imputed DataFrame (same shape/columns as X_missing).
            models_: Empty dict (HyperImpute manages models internally).
        """
        try:
            from hyperimpute.plugins.imputers import Imputers
        except ImportError as exc:
            raise ImportError(
                "HyperImpute requires the 'hyperimpute' package. "
                "Install it with: pip install hyperimpute"
            ) from exc

        df = X_missing.copy()

        # --- Encode categorical columns to integer codes ---
        category_mappings: Dict[str, list] = {}
        for col in self.categorical_vars:
            if col not in df.columns:
                continue
            # R0: build categories from complete data to ensure full label set.
            # R1 (X_complete is None): from the observed cells of X_missing, or
            # from the fitted vocabulary installed by the registry.
            if X_complete is not None and col in X_complete.columns:
                all_cats = X_complete[col].dropna().unique().tolist()
                try:
                    all_cats = sorted(all_cats)
                except Exception:
                    pass
            elif isinstance(df[col].dtype, pd.CategoricalDtype):
                all_cats = list(df[col].cat.categories)
            else:
                all_cats = df[col].dropna().unique().tolist()
                try:
                    all_cats = sorted(all_cats)
                except Exception:
                    pass

            category_mappings[col] = all_cats
            cat_type = pd.CategoricalDtype(categories=all_cats)
            df[col] = df[col].astype(cat_type).cat.codes.astype(float)
            # cat.codes returns -1 for NaN — convert back to NaN
            df.loc[df[col] < 0, col] = np.nan

        # --- Ensure continuous columns are numeric ---
        for col in self.continuous_vars:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Set random seeds ---
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass

        # --- Debug: list available imputer plugins ---
        imputers_api = Imputers()
        try:
            available = imputers_api.list()
            log.debug("HyperImpute available imputer plugins: %s", available)
        except Exception:
            log.debug("HyperImpute: could not list available plugins")

        # --- Build HyperImpute plugin ---
        # Only pass classifier_seed / regression_seed when the caller
        # supplied them explicitly.  When they are None the package
        # falls back to its own built-in defaults (the safest option,
        # because plugin names can differ across hyperimpute versions).
        plugin_kwargs: Dict[str, object] = {
            "optimizer": self.optimizer,
            "random_state": self.seed,
        }
        if self.classifier_seed is not None:
            plugin_kwargs["classifier_seed"] = self.classifier_seed
        if self.regression_seed is not None:
            plugin_kwargs["regression_seed"] = self.regression_seed

        log.debug("HyperImpute plugin kwargs: %s", plugin_kwargs)
        plugin = imputers_api.get("hyperimpute", **plugin_kwargs)

        # --- Run with timeout ---
        df_imputed_values = self._run_with_timeout(plugin, df)

        # --- Build output DataFrame ---
        df_imputed = pd.DataFrame(
            df_imputed_values,
            columns=df.columns,
            index=df.index,
        )

        # --- Decode categorical columns back to original labels ---
        for col in self.categorical_vars:
            if col not in df_imputed.columns:
                continue
            if col not in category_mappings:
                continue

            original_categories = category_mappings[col]
            max_code = len(original_categories) - 1

            # Round to nearest integer code and clip
            codes = df_imputed[col].values.astype(float)
            codes = np.round(codes).astype(int)
            codes = np.clip(codes, 0, max_code)

            decoded = [original_categories[int(c)] for c in codes]
            df_imputed[col] = decoded

        # --- Ensure continuous columns are float ---
        for col in self.continuous_vars:
            if col in df_imputed.columns:
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

        return df_imputed, self.models_

    # ------------------------------------------------------------------
    # R1 additions: fit / transform
    # ------------------------------------------------------------------
    def _encode(self, X, category_mappings=None):
        df = X.copy()
        mappings = {} if category_mappings is None else category_mappings
        for col in self.categorical_vars:
            if col not in df.columns:
                continue
            if category_mappings is None:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    all_cats = list(df[col].cat.categories)
                else:
                    all_cats = df[col].dropna().unique().tolist()
                    try:
                        all_cats = sorted(all_cats)
                    except Exception:
                        pass
                mappings[col] = all_cats
            else:
                all_cats = mappings[col]
            df[col] = df[col].astype(pd.CategoricalDtype(categories=all_cats)).cat.codes.astype(float)
            df.loc[df[col] < 0, col] = np.nan
        for col in self.continuous_vars:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df, mappings

    def _decode(self, values, columns, index, category_mappings):
        df_imputed = pd.DataFrame(values, columns=columns, index=index)
        for col in self.categorical_vars:
            if col not in df_imputed.columns or col not in category_mappings:
                continue
            cats = category_mappings[col]
            codes = np.clip(np.round(df_imputed[col].values.astype(float)).astype(int),
                            0, len(cats) - 1)
            df_imputed[col] = [cats[int(c)] for c in codes]
        for col in self.continuous_vars:
            if col in df_imputed.columns:
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")
        return df_imputed

    def fit(self, X_train_missing, stats=None):
        """DOES NOT PROVIDE fit/transform SEPARATION. Evidence-only entry point.

        ``HyperImputePlugin`` inherits ``sklearn.impute._base._BaseImputer`` and
        advertises ``fit`` / ``transform``, but in the installed package the pair
        is a facade::

            def _fit(self, X, *args, **kwargs):     # plugin_hyperimpute.py:124-125
                return self

            def _transform(self, X):                # plugin_hyperimpute.py:127-128
                return self.model.fit_transform(X)

        ``fit`` stores nothing and ``transform`` reruns the entire AutoML search
        on whatever frame it is handed. Verified empirically: after
        ``p.fit(train)``, ``p.transform(test)`` is elementwise identical to a
        fresh plugin's ``fit_transform(test)`` -- the training fold has no
        influence whatsoever.

        These methods are kept so ``tests/test_baselines.py`` can assert that
        property (and detect a future package version that fixes it), but the
        registry deliberately reports ``supports_fit_transform = False`` for
        HyperImpute and routes it to the fold-independent protocol.
        """
        from hyperimpute.plugins.imputers import Imputers

        df, mappings = self._encode(X_train_missing, None)
        self.category_mappings_ = mappings
        self.stats_ = stats

        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass

        plugin_kwargs: Dict[str, object] = {
            "optimizer": self.optimizer,
            "random_state": self.seed,
        }
        if self.classifier_seed is not None:
            plugin_kwargs["classifier_seed"] = self.classifier_seed
        if self.regression_seed is not None:
            plugin_kwargs["regression_seed"] = self.regression_seed

        plugin = Imputers().get("hyperimpute", **plugin_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plugin.fit(df)
        self.plugin_ = plugin
        return self

    def transform(self, X_new_missing):
        if getattr(self, "plugin_", None) is None:
            raise RuntimeError("HyperImputeImputer.transform called before fit")
        df, _ = self._encode(X_new_missing, self.category_mappings_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.plugin_.transform(df)
        values = out.values if hasattr(out, "values") else np.asarray(out)
        return self._decode(values, df.columns, df.index, self.category_mappings_), self.models_

    def _run_with_timeout(self, plugin, df: pd.DataFrame) -> np.ndarray:
        """Run HyperImpute with a timeout guard."""

        class TimeoutError(Exception):
            pass

        def _handler(signum, frame):
            raise TimeoutError(
                f"HyperImpute exceeded timeout of {self.timeout}s"
            )

        # Try signal-based timeout (Unix only)
        use_signal = hasattr(signal, "SIGALRM")

        if use_signal:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.timeout)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                result = plugin.fit_transform(df)

            if hasattr(result, "values"):
                return result.values
            return np.asarray(result)

        except TimeoutError:
            warnings.warn(
                f"HyperImpute timed out after {self.timeout}s. "
                "Returning partially imputed result with mean/mode fallback.",
                RuntimeWarning,
            )
            # Fallback: fill remaining NaN with mean/mode
            df_fallback = df.copy()
            for col in df_fallback.columns:
                if df_fallback[col].isna().any():
                    if col in self.categorical_vars:
                        mode_val = df_fallback[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    else:
                        fill_val = df_fallback[col].mean(skipna=True)
                        if pd.isna(fill_val):
                            fill_val = 0.0
                    df_fallback[col] = df_fallback[col].fillna(fill_val)
            return df_fallback.values

        except Exception as exc:
            warnings.warn(
                f"HyperImpute failed: {exc}. "
                "Returning mean/mode fallback imputation.",
                RuntimeWarning,
            )
            df_fallback = df.copy()
            for col in df_fallback.columns:
                if df_fallback[col].isna().any():
                    if col in self.categorical_vars:
                        mode_val = df_fallback[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    else:
                        fill_val = df_fallback[col].mean(skipna=True)
                        if pd.isna(fill_val):
                            fill_val = 0.0
                    df_fallback[col] = df_fallback[col].fillna(fill_val)
            return df_fallback.values

        finally:
            if use_signal:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


# Backward-compatible alias
hyperImputeImputer = HyperImputeImputer


def _run_self_test() -> None:
    """Quick synthetic data self-test (no CLI args)."""
    print("HyperImpute v1 self-test")
    print("=" * 60)

    np.random.seed(42)
    n = 200

    x1 = np.random.randn(n) * 10 + 50
    x2 = np.random.randn(n) * 5 + 20
    x3 = x1 * 0.5 + x2 * 0.3 + np.random.randn(n) * 2

    cat1 = np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(["X", "Y"], n)

    df_complete = pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "cat1": cat1, "cat2": cat2}
    )

    df_missing = df_complete.copy()
    miss_rate = 0.2

    for col in df_missing.columns:
        miss_idx = np.random.choice(n, size=int(n * miss_rate), replace=False)
        df_missing.loc[miss_idx, col] = np.nan

    print(f"Missing rate: {df_missing.isna().mean().mean():.1%}")
    print(f"Per-column NaN count: {df_missing.isna().sum().to_dict()}")

    imputer = HyperImputeImputer(
        categorical_vars=["cat1", "cat2"],
        continuous_vars=["x1", "x2", "x3"],
        seed=42,
        timeout=120,
    )

    df_imputed, _ = imputer.impute(df_complete, df_missing)

    print("\n" + "=" * 60)
    print("Imputation evaluation")
    print("=" * 60)

    for col in ["x1", "x2", "x3"]:
        mask = df_missing[col].isna()
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = pd.to_numeric(df_imputed.loc[mask, col], errors="coerce").values
            rmse = np.sqrt(np.mean((true_vals - imp_vals) ** 2))
            nrmse_val = rmse / np.std(true_vals) if np.std(true_vals) > 0 else rmse
            print(f"  {col}: NRMSE = {nrmse_val:.4f}")

    for col in ["cat1", "cat2"]:
        mask = df_missing[col].isna()
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = df_imputed.loc[mask, col].values
            pfc = np.mean(true_vals != imp_vals)
            print(f"  {col}: PFC = {pfc:.4f}")


def _run_cli() -> None:
    """CLI mode: read data, impute, compute metrics, write outputs."""
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="HyperImpute CLI — impute missing data and write metrics_summary.json"
    )
    ap.add_argument("--input-complete", required=True, help="Path to complete CSV")
    ap.add_argument("--input-missing", required=True, help="Path to missing CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--categorical-vars", type=str, default="", help="Comma-separated categorical column names")
    ap.add_argument("--continuous-vars", type=str, default="", help="Comma-separated continuous column names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default 600)")
    ap.add_argument("--optimizer", type=str, default="hyperband")
    ap.add_argument("--exp-id", type=str, default="", help="Experiment ID for metrics_summary.json")
    args = ap.parse_args()

    cat_vars = [v.strip() for v in args.categorical_vars.split(",") if v.strip()]
    cont_vars = [v.strip() for v in args.continuous_vars.split(",") if v.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_complete = pd.read_csv(args.input_complete)
    X_missing = pd.read_csv(args.input_missing)

    # Align columns
    all_vars = cat_vars + cont_vars
    if all_vars:
        X_complete = X_complete[all_vars]
        X_missing = X_missing[all_vars]

    print(f"[HyperImpute CLI] complete shape: {X_complete.shape}, missing shape: {X_missing.shape}")
    print(f"[HyperImpute CLI] categorical_vars={cat_vars}, continuous_vars={cont_vars}")

    t0 = time.time()

    imputer = HyperImputeImputer(
        categorical_vars=cat_vars,
        continuous_vars=cont_vars,
        seed=args.seed,
        timeout=args.timeout,
        optimizer=args.optimizer,
    )

    df_imputed, _ = imputer.impute(X_complete, X_missing)
    runtime_sec = time.time() - t0

    # Save imputed data
    df_imputed.to_csv(outdir / "imputed.csv", index=False)

    # Compute metrics using SNI_v0_3.metrics (same as run_manifest_baselines.py)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from SNI_v0_3.metrics import evaluate_imputation

        eval_res = evaluate_imputation(
            df_imputed, X_complete, X_missing,
            categorical_vars=cat_vars,
            continuous_vars=cont_vars,
        )
        summary = dict(eval_res.summary)
        eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)
    except ImportError:
        # Fallback: compute basic metrics inline
        summary = {}
        for col in cont_vars:
            mask = X_missing[col].isna()
            if mask.sum() > 0:
                true_vals = pd.to_numeric(X_complete.loc[mask, col], errors="coerce").values
                imp_vals = pd.to_numeric(df_imputed.loc[mask, col], errors="coerce").values
                valid = np.isfinite(true_vals) & np.isfinite(imp_vals)
                if valid.sum() > 0:
                    rmse = float(np.sqrt(np.mean((true_vals[valid] - imp_vals[valid]) ** 2)))
                    vr = float(np.nanmax(true_vals[valid]) - np.nanmin(true_vals[valid]))
                    summary.setdefault("cont_NRMSE", []).append(rmse / vr if vr > 0 else rmse)  # type: ignore[union-attr]
        if "cont_NRMSE" in summary and isinstance(summary["cont_NRMSE"], list):
            summary["cont_NRMSE"] = float(np.mean(summary["cont_NRMSE"]))

    summary.update({
        "exp_id": args.exp_id or Path(args.outdir).name,
        "method": "HyperImpute",
        "seed": args.seed,
        "use_gpu": False,
        "runtime_sec": round(runtime_sec, 3),
    })

    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[HyperImpute CLI] Done in {runtime_sec:.1f}s → {outdir / 'metrics_summary.json'}")
    for k, v in summary.items():
        if k.startswith("cont_") or k.startswith("cat_"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    # If CLI args are provided (beyond script name), use CLI mode; otherwise self-test.
    if len(sys.argv) > 1:
        _run_cli()
    else:
        _run_self_test()
