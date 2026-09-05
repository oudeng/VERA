# imputer.py - SNI Imputer (v0.3)
# Statistical-Neural Interaction for Missing Data Imputation
# v0.3: categorical balance, lambda ablation, convergence monitoring

from __future__ import annotations

import json
import time

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .cpfa import EnhancedCPFA, DualPathCPFA, CategoricalCPFATrainer
from .dataio import cast_dataframe_to_schema
from .utils import set_global_seed, DeviceConfig, enable_performance_mode

import warnings
warnings.filterwarnings(
    'ignore', 
    message='Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`'
)

def categorical_aware_pseudo_masking(y: np.ndarray, mask_rate: float = 0.15) -> np.ndarray:
    """
    Stratified pseudo-masking for categorical labels to preserve class distribution.
    """
    y = np.asarray(y)
    uniq, counts = np.unique(y, return_counts=True)
    n = len(y)
    freqs = counts / max(1, n)

    class_mask_rates: Dict[object, float] = {}
    for cls, freq in zip(uniq, freqs):
        if freq < 0.01:
            class_mask_rates[cls] = max(0.05, mask_rate * 0.3)
        elif freq < 0.05:
            class_mask_rates[cls] = mask_rate * 0.7
        else:
            class_mask_rates[cls] = mask_rate

    mask = np.zeros(n, dtype=bool)
    for cls in uniq:
        cls_idx = np.where(y == cls)[0]
        n_mask = int(len(cls_idx) * class_mask_rates[cls])
        n_mask = min(n_mask, max(0, len(cls_idx) - 2))
        if n_mask > 0:
            sel = np.random.choice(cls_idx, n_mask, replace=False)
            mask[sel] = True
    return mask


def _encode_categorical_for_mice(x: pd.Series) -> Tuple[pd.Series, Dict[int, object]]:
    """
    Encode categorical series to integer codes for MICE.
    """
    s = x.copy()
    mask_na = s.isna()
    s_non_na = s[~mask_na].astype(object)

    uniq = pd.unique(s_non_na)
    code_map = {v: i for i, v in enumerate(uniq)}
    inv_map = {i: v for v, i in code_map.items()}

    encoded = s.astype(object).map(code_map).astype(float)
    encoded[mask_na] = np.nan
    return encoded, inv_map


def _decode_categorical_from_mice(x_num: np.ndarray, inv_map: Dict[int, object]) -> np.ndarray:
    if len(inv_map) == 0:
        return x_num
    k = len(inv_map)
    x_round = np.round(x_num).astype(int)
    x_round = np.clip(x_round, 0, k - 1)
    return np.array([inv_map[int(v)] for v in x_round], dtype=object)


@dataclass
class SNIConfig:
    """
    SNI Configuration - High-Performance defaults for v0.3
    
    Optimized for maximum imputation accuracy (SOTA performance).
    
    Key design choices:
    - hidden_dims: (256, 128, 64) - 3-layer MLP for maximum capacity
    - emb_dim: 128 - Rich feature embeddings
    - num_heads: 16 - Diverse attention patterns
    - epochs: 200 - Sufficient training with early stopping
    - lr: 2e-4 - Conservative learning rate for stability
    - batch_size: 128 - Stable gradient estimates
    - gamma: 0.9 - Balanced prior decay (neural-network-led)
    - mask_fraction: 0.15 - Strong pseudo-supervision
    """
    # EM / prior
    alpha0: float = 1.0
    gamma: float = 0.9              # Prior decay (0.9 lets NN dominate faster)
    max_iters: int = 3              # EM iterations
    tol: float = 1e-4
    use_stat_refine: bool = True

    # Pseudo mask
    mask_fraction: float = 0.15     # Increased pseudo-supervision

    # CPFA architecture (High-Capacity)
    hidden_dims: Tuple[int, ...] = (256, 128, 64)  # 3-layer MLP
    emb_dim: int = 128                             # Rich embeddings
    num_heads: int = 16                            # Many attention patterns
    
    # Training (Performance-Optimized)
    lr: float = 2e-4                            # Conservative LR
    weight_decay: float = 1e-4                  # Moderate regularization
    epochs: int = 200                           # Full training
    batch_size: int = 128                       # Stable gradients
    early_stopping_patience: int = 20           # Patient early stopping
    min_epochs: int = 50                        # Ensure sufficient training

    # Categorical training techniques
    use_dual_path: bool = True
    use_multiscale: bool = True
    use_cat_embedding: bool = True
    label_smoothing_epsilon: float = 0.1        # Standard smoothing
    use_focal_loss: bool = True
    use_label_smoothing: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.2                    # Mixup strength

    # v0.3: Categorical balance & LR multiplier
    cat_balance_mode: str = "none"              # 'none' | 'inverse_freq' | 'sqrt_inverse_freq'
    cat_lr_mult: float = 1.0                    # LR multiplier for categorical heads (default 1.0 = no change)

    # Variants: {SNI, NoPrior, HardPrior, SNI-M, SNI+KNN}
    variant: str = "SNI"
    hard_prior_lambda: float = 10.0

    # v0.3: Lambda ablation
    lambda_mode: str = "learned"                # 'learned' (default) | 'fixed'
    lambda_fixed_value: float = 1.0             # Used when lambda_mode='fixed'

    # Performance
    seed: int = 2025
    use_gpu: bool = False
    num_workers: int = 0            # DataLoader workers (0 = main thread)
    pin_memory: bool = True         # For GPU transfers

    # --- code_SNI E3 (P1/T1.3) --------------------------------------------
    # R0 hardcoded `if use_gpu: enable_performance_mode()` in impute(), which
    # reversed the cudnn determinism flags set moments earlier by
    # set_global_seed() and switched TF32 on. Since every SNI manifest had
    # use_gpu=True, no published SNI number is bit-reproducible (P0 finding B10).
    #
    # The policy is now an explicit, recorded parameter. Nothing else changed:
    #   determinism_mode="performance"   -> byte-for-byte the R0 behavior
    #   determinism_mode="deterministic" -> reproducible (the code_SNI default)
    # The equivalence check in T1.3 runs both so the two can be compared.
    determinism_mode: str = "deterministic"   # 'deterministic' | 'performance'


class SNIImputer:
    """
    Statistical-Neural Interaction (SNI) Imputer - v0.3
    
    A hybrid imputation method combining:
    1. Statistical priors from correlation analysis
    2. Neural attention for flexible feature dependencies
    3. Learnable confidence parameters for prior-data balance
    
    Key innovations:
    - CPFA (Controllable-Prior Feature Attention) architecture
    - Per-head learnable λ for adaptive prior strength
    - Multi-scale attention for categorical features
    - EM-style iterative refinement
    """
    
    def __init__(self, categorical_vars: List[str], continuous_vars: List[str], config: Optional[SNIConfig] = None):
        self.cat_vars = list(categorical_vars)
        self.cont_vars = list(continuous_vars)
        self.all_vars = self.cat_vars + self.cont_vars

        self.cfg = config or SNIConfig()

        self.device = DeviceConfig(use_gpu=self.cfg.use_gpu).torch_device()

        # Encoders for categorical features
        self.encoders: Dict[str, LabelEncoder] = {v: LabelEncoder() for v in self.cat_vars}
        
        # Feature encoders for training (stores mapping from string to int for each categorical column)
        self._feature_encoders: Dict[str, Dict[object, int]] = {}

        # Trainer for categorical CPFA
        self.cat_trainer = CategoricalCPFATrainer(
            use_focal_loss=self.cfg.use_focal_loss,
            use_label_smoothing=self.cfg.use_label_smoothing,
            use_mixup=self.cfg.use_mixup,
            mixup_alpha=0.2,
            use_curriculum=True,
        )

        # Artifacts
        self.models: Dict[str, nn.Module] = {}
        self.attention_maps: Dict[str, np.ndarray] = {}
        self.lambda_trace_per_head: Dict[str, List[List[float]]] = {}
        self.logs: Dict[str, Dict[str, List[float]]] = {}

        # Dependency matrix (computed on demand)
        self.dependency_matrix_: Optional[pd.DataFrame] = None

        # v0.3: Convergence tracking & timing
        self.convergence_curve_: List[Dict[str, float]] = []
        self.convergence_iterations_: int = 0
        self.runtime_sec_: float = 0.0
        self.did_converge_: bool = False

    def impute(
        self,
        X_missing: pd.DataFrame,
        X_complete: Optional[pd.DataFrame] = None,
        mask_df: Optional[pd.DataFrame] = None,
        return_artifacts: bool = True,
    ) -> pd.DataFrame:
        """
        Run SNI imputation.
        
        Args:
            X_missing: DataFrame with NaNs to impute
            X_complete: Optional ground truth (for evaluation only)
            mask_df: Optional explicit missingness mask
            return_artifacts: Whether to store attention maps etc.
            
        Returns:
            Imputed DataFrame
        """
        # code_SNI E3: seeding and determinism policy are applied together and the
        # resolved state is kept on the instance so the runner can record it in
        # run_config.json. See the determinism_mode field of SNIConfig.
        from common import determinism as _determinism

        self.determinism_state_ = _determinism.apply(
            self.cfg.determinism_mode,
            seed=self.cfg.seed,
        )
        _t0 = time.time()

        X_missing = X_missing.copy()

        # Ensure column order
        X_missing = X_missing[self.all_vars]
        if X_complete is not None:
            X_complete = X_complete[self.all_vars]

        # Build schema
        schema: Dict[str, str] = {}
        for c in self.cont_vars:
            schema[c] = "float64"
        for c in self.cat_vars:
            dt = X_missing[c].dtype
            if pd.api.types.is_numeric_dtype(dt) or str(dt) == "Int64":
                schema[c] = "Int64"
            else:
                schema[c] = "category"

        X_missing = cast_dataframe_to_schema(X_missing, schema)
        if X_complete is not None:
            X_complete = cast_dataframe_to_schema(X_complete, schema)

        if mask_df is None:
            mask_df = X_missing.isna()
        else:
            mask_df = mask_df[self.all_vars]

        def _assign_preds(df: pd.DataFrame, col: str, idx: np.ndarray, preds: np.ndarray) -> None:
            """Assign predictions with dtype safety."""
            if len(idx) == 0:
                return
            tgt = df[col]
            tgt_dtype = tgt.dtype
            vals = np.asarray(preds)

            if pd.api.types.is_integer_dtype(tgt_dtype) or str(tgt_dtype) == "Int64":
                v = pd.to_numeric(pd.Series(vals), errors="coerce").round().astype("Int64")
                df.loc[df.index[idx], col] = v.values
            elif pd.api.types.is_float_dtype(tgt_dtype):
                v = pd.to_numeric(pd.Series(vals), errors="coerce").astype("float64")
                df.loc[df.index[idx], col] = v.values
            else:
                df.loc[df.index[idx], col] = vals.astype(object)

        # Initial statistical imputation
        X_current = self._initial_stat_impute(X_missing)
        X_current = cast_dataframe_to_schema(X_current, schema)

        # v0.3: Reset convergence tracking
        self.convergence_curve_ = []
        self.did_converge_ = False

        # EM loop
        for g in range(1, self.cfg.max_iters + 1):
            alpha = self.cfg.alpha0 * (self.cfg.gamma ** (g - 1))

            # Compute priors from current completion
            Prior_matrix, cat_onehot_dims, corr_cols = self._compute_correlation_prior(X_current)

            X_next = X_current.copy()

            # v0.3: Track per-iteration losses
            cont_losses_iter: List[float] = []
            cat_losses_iter: List[float] = []

            for f in self.all_vars:
                P_f = self._extract_feature_prior(Prior_matrix, f, corr_cols, cat_onehot_dims)
                P_f = self._normalize_prior(P_f)

                if self.cfg.variant == "NoPrior":
                    alpha_f = 0.0
                else:
                    alpha_f = alpha

                # Train CPFA for this feature
                if f in self.cont_vars:
                    model_f, preds_missing = self._train_continuous_feature(
                        f, X_current, X_missing, mask_df, P_f, alpha_f
                    )
                    miss_idx = np.where(mask_df[f].values)[0]
                    if len(miss_idx) > 0:
                        _assign_preds(X_next, f, miss_idx, preds_missing)
                    # Record final training loss for convergence curve
                    if f in self.logs and len(self.logs[f]["tr"]) > 0:
                        cont_losses_iter.append(self.logs[f]["tr"][-1])
                else:
                    model_f, preds_missing = self._train_categorical_feature(
                        f, X_current, X_missing, mask_df, P_f, alpha_f
                    )
                    miss_idx = np.where(mask_df[f].values)[0]
                    if len(miss_idx) > 0:
                        _assign_preds(X_next, f, miss_idx, preds_missing)
                    # Record final training loss for convergence curve
                    if f in self.logs and len(self.logs[f]["tr"]) > 0:
                        cat_losses_iter.append(self.logs[f]["tr"][-1])

                self.models[f] = model_f

            # Statistical refinement
            if self.cfg.use_stat_refine:
                X_refined = self._stat_refine(X_next)
                X_refined = cast_dataframe_to_schema(X_refined, schema)
            else:
                X_refined = X_next

            # Convergence check
            delta = self._max_imputed_delta(X_current, X_refined, mask_df)
            X_current = X_refined

            # v0.3: Record convergence curve entry
            cont_loss_avg = float(np.mean(cont_losses_iter)) if cont_losses_iter else 0.0
            cat_loss_avg = float(np.mean(cat_losses_iter)) if cat_losses_iter else 0.0
            self.convergence_curve_.append({
                "iteration": g,
                "cont_loss": cont_loss_avg,
                "cat_loss": cat_loss_avg,
                "delta": delta,
            })

            if delta < self.cfg.tol:
                self.did_converge_ = True
                self.convergence_iterations_ = g
                break
        else:
            # Loop completed without break → did not converge
            self.did_converge_ = False
            self.convergence_iterations_ = self.cfg.max_iters

        # v0.3: Record timing
        self.runtime_sec_ = time.time() - _t0

        # Gate-bound runtime assertion (P5R-C SS3.4 / review Q10): with
        # learned lambda, softplus(theta) <= ln 2 is an optimization
        # invariant (theta starts at 0 and the gate enters the objective
        # only through the non-negative prior term, so every gradient
        # step is downhill in theta; decoupled decay cannot cross zero).
        # Checked here on every trained host; derivation in the ESM.
        if self.cfg.lambda_mode == "learned":
            import math

            import torch.nn.functional as _F
            for _f, _m in self.models.items():
                if hasattr(_m, "theta_lambda"):
                    _lmax = float(_F.softplus(_m.theta_lambda).max())
                    assert _lmax <= math.log(2.0) + 1e-6, (
                        f"lambda invariant violated for {_f}: "
                        f"max lambda {_lmax:.6f} > ln 2")

        return X_current

    def compute_dependency_matrix(self) -> pd.DataFrame:
        """
        Build dependency matrix D from attention maps.
        D[i,j] = importance of feature j for predicting feature i.
        """
        if self.dependency_matrix_ is not None:
            return self.dependency_matrix_

        features = self.all_vars
        d = len(features)
        D = np.zeros((d, d), dtype=float)

        for i, target in enumerate(features):
            if target not in self.attention_maps:
                continue
            attn_map = self.attention_maps[target]
            avg_attention = attn_map.mean(axis=0)
            other_features = [f for f in features if f != target]

            for j, src in enumerate(features):
                if src == target:
                    D[i, j] = 0.0
                else:
                    idx = other_features.index(src)
                    D[i, j] = float(avg_attention[idx])

        # Row normalize
        row_sums = D.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        D = D / row_sums

        df = pd.DataFrame(D, index=features, columns=features)
        self.dependency_matrix_ = df
        return df

    def export_dependency_network_edges(self, tau: float = 0.15) -> pd.DataFrame:
        """
        Convert dependency matrix to edge list for visualization.
        """
        D = self.compute_dependency_matrix()
        edges = []
        for target in D.index:
            for src in D.columns:
                if target == src:
                    continue
                w = float(D.loc[target, src])
                if w > tau:
                    edges.append({"target": target, "source": src, "weight": w})
        return pd.DataFrame(edges)

    # -------------------------
    # v0.3: Output helpers
    # -------------------------
    def save_convergence_curve(self, path: str) -> None:
        """Save convergence curve to CSV (iteration, cont_loss, cat_loss, delta)."""
        if self.convergence_curve_:
            pd.DataFrame(self.convergence_curve_).to_csv(path, index=False)

    def get_lambda_per_head_df(self) -> pd.DataFrame:
        """Return DataFrame of final λ values per feature per head.

        When ``lambda_mode='fixed'``, every head reports ``lambda_fixed_value``
        (the value actually used during training), instead of the raw
        ``softplus(theta_lambda)`` which is never used in fixed mode.
        """
        rows = []
        for f in self.all_vars:
            if f not in self.models:
                continue
            model = self.models[f]
            if hasattr(model, "theta_lambda"):
                if self.cfg.lambda_mode == "fixed":
                    n_heads = model.theta_lambda.numel()
                    lambdas = np.full(n_heads, self.cfg.lambda_fixed_value)
                else:
                    import torch.nn.functional as _F
                    lambdas = _F.softplus(model.theta_lambda).detach().cpu().numpy()
                row = {"feature": f}
                for h, lv in enumerate(lambdas):
                    row[f"head_{h}"] = float(lv)
                rows.append(row)
        return pd.DataFrame(rows)

    def save_lambda_per_head(self, path: str) -> None:
        """Save lambda_per_head.csv."""
        df = self.get_lambda_per_head_df()
        if not df.empty:
            df.to_csv(path, index=False)

    def get_final_lambda_values(self) -> Dict[str, List[float]]:
        """Return final λ_h values for each feature (for lambda_values.json).

        When ``lambda_mode='fixed'``, returns the fixed value for every head,
        matching the value actually used during training.
        """
        result: Dict[str, List[float]] = {}
        for f in self.all_vars:
            if f not in self.models:
                continue
            model = self.models[f]
            if hasattr(model, "theta_lambda"):
                if self.cfg.lambda_mode == "fixed":
                    n_heads = model.theta_lambda.numel()
                    lambdas = [float(self.cfg.lambda_fixed_value)] * n_heads
                else:
                    import torch.nn.functional as _F
                    lambdas = _F.softplus(model.theta_lambda).detach().cpu().numpy().tolist()
                result[f] = lambdas
        return result

    def save_lambda_values(self, path: str) -> None:
        """Save lambda_values.json with final λ_h per feature."""
        lv = self.get_final_lambda_values()
        with open(path, "w") as fp:
            json.dump(lv, fp, indent=2)

    def get_lambda_summary(self) -> Dict[str, float]:
        """Return lambda_mean and lambda_std across all heads of all features."""
        all_lambdas: List[float] = []
        for vals in self.get_final_lambda_values().values():
            all_lambdas.extend(vals)
        if all_lambdas:
            arr = np.array(all_lambdas)
            return {"lambda_mean": float(arr.mean()), "lambda_std": float(arr.std())}
        return {"lambda_mean": 0.0, "lambda_std": 0.0}

    # -------------------------
    # Internal: initialization
    # -------------------------
    def _initial_stat_impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        """
        Initial statistical imputation via MICE.
        """
        X = X_missing.copy()

        inv_maps: Dict[str, Dict[int, object]] = {}
        for col in self.cat_vars:
            if col not in X.columns:
                continue
            enc, inv_map = _encode_categorical_for_mice(X[col])
            X[col] = enc
            inv_maps[col] = inv_map

        X_num = X.apply(pd.to_numeric, errors="coerce").values.astype(float)

        imputer = IterativeImputer(
            random_state=self.cfg.seed,
            max_iter=5,              # Reduced from 10
            sample_posterior=False,
            initial_strategy="mean",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            X_imp = imputer.fit_transform(X_num)
        X_imp_df = pd.DataFrame(X_imp, columns=X.columns, index=X.index)

        for col in self.cat_vars:
            inv_map = inv_maps.get(col, {})
            decoded = _decode_categorical_from_mice(X_imp_df[col].values, inv_map)
            X_imp_df[col] = decoded

        return X_imp_df

    def _stat_refine(self, X_partial: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical refinement using MICE.
        """
        X = X_partial.copy()

        inv_maps: Dict[str, Dict[int, object]] = {}
        for col in self.cat_vars:
            if col not in X.columns:
                continue
            enc, inv_map = _encode_categorical_for_mice(X[col])
            X[col] = enc
            inv_maps[col] = inv_map

        X_num = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
        imputer = IterativeImputer(
            random_state=self.cfg.seed,
            max_iter=3,              # Reduced from 5
            sample_posterior=False,
            initial_strategy="mean",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            X_ref = imputer.fit_transform(X_num)
        X_ref_df = pd.DataFrame(X_ref, columns=X.columns, index=X.index)

        for col in self.cat_vars:
            inv_map = inv_maps.get(col, {})
            if col in X_ref_df.columns:
                X_ref_df[col] = _decode_categorical_from_mice(X_ref_df[col].values, inv_map)

        return X_ref_df

    def _compute_correlation_prior(self, X_full: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, List[str]], List[str]]:
        """
        Compute correlation-based prior matrix.
        """
        blocks = []
        col_names: List[str] = []
        cat_onehot_dims: Dict[str, List[str]] = {}

        for col in self.all_vars:
            if col in self.cat_vars:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                try:
                    arr = X_full[col].astype(str).values.reshape(-1, 1)
                    ohe.fit(arr)
                    cat_names = [f"{col}__{c}" for c in ohe.categories_[0]]
                    col_names.extend(cat_names)
                    cat_onehot_dims[col] = cat_names
                    blocks.append(ohe.transform(arr))
                except Exception:
                    col_names.append(col)
                    cat_onehot_dims[col] = [col]
                    blocks.append(X_full[[col]].apply(pd.to_numeric, errors="coerce").values)
            else:
                col_names.append(col)
                blocks.append(X_full[[col]].apply(pd.to_numeric, errors="coerce").values)

        X_enc = np.hstack(blocks)
        # Handle NaN for correlation
        X_enc = np.nan_to_num(X_enc, nan=0.0)
        
        # Pearson correlation
        if X_enc.shape[1] > 1:
            corr_mat = np.corrcoef(X_enc, rowvar=False)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        else:
            corr_mat = np.ones((1, 1))
        
        # Convert to absolute values for prior
        Prior_matrix = np.abs(corr_mat)
        
        return Prior_matrix, cat_onehot_dims, col_names

    def _extract_feature_prior(
        self,
        Prior_matrix: np.ndarray,
        target_feature: str,
        corr_cols: List[str],
        cat_onehot_dims: Dict[str, List[str]],
    ) -> np.ndarray:
        """
        Extract prior vector for a target feature from correlation matrix.
        """
        # Get indices for target feature
        if target_feature in cat_onehot_dims:
            idxs_f = [corr_cols.index(c) for c in cat_onehot_dims[target_feature]]
        else:
            idxs_f = [corr_cols.index(target_feature)]

        P_list = []
        for var2 in self.all_vars:
            if var2 == target_feature:
                continue
            if var2 not in cat_onehot_dims:
                j2 = corr_cols.index(var2)
                corr_vals = [Prior_matrix[i_f, j2] for i_f in idxs_f]
                P_list.append(float(np.mean(corr_vals)))
            else:
                idxs2 = [corr_cols.index(c) for c in cat_onehot_dims[var2]]
                corr_vals = []
                for i_f in idxs_f:
                    for j2 in idxs2:
                        corr_vals.append(Prior_matrix[i_f, j2])
                P_list.append(float(np.mean(corr_vals)))

        return np.asarray(P_list, dtype=float)

    @staticmethod
    def _normalize_prior(P_f: np.ndarray) -> np.ndarray:
        P = np.asarray(P_f, dtype=float)
        P[P < 0] = 0.0
        s = float(P.sum())
        if s <= 0:
            return np.ones_like(P) / max(1, len(P))
        return P / s

    def _get_categorical_indices_excluding_target(self, target: str, all_features: List[str]) -> List[int]:
        """Get indices of categorical tokens in Z (excluding target)."""
        # Build the feature list excluding target
        features_excl_target = [v for v in all_features if v != target]
        cat_idx = []
        for i, v in enumerate(features_excl_target):
            if v in self.cat_vars:
                cat_idx.append(i)
        return cat_idx

    @staticmethod
    def _max_imputed_delta(X_old: pd.DataFrame, X_new: pd.DataFrame, mask_df: pd.DataFrame, cat_vars: List[str] = None) -> float:
        deltas = []
        for col in X_old.columns:
            miss = mask_df[col]
            if int(miss.sum()) == 0:
                continue
            
            if cat_vars and col in cat_vars:
                # カテゴリ変数: 変化した値の割合
                changed = (X_old.loc[miss, col].astype(str) != X_new.loc[miss, col].astype(str))
                deltas.append(float(changed.mean()))
            else:
                # 連続変数: 最大差分
                old_vals = pd.to_numeric(X_old.loc[miss, col], errors="coerce")
                new_vals = pd.to_numeric(X_new.loc[miss, col], errors="coerce")
                diff = (new_vals - old_vals).abs()
                deltas.append(float(diff.max()))
        
        return float(np.nanmax(deltas)) if len(deltas) > 0 else 0.0

    # -------------------------
    # NEW: Encode DataFrame for training
    # -------------------------
    def _encode_dataframe_for_training(self, Z_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Dict[object, int]]]:
        """
        Encode a DataFrame for neural network training.
        Categorical columns are encoded to integer codes, continuous columns are kept as-is.
        
        Args:
            Z_df: DataFrame with features (may include categorical string columns)
            
        Returns:
            Z_encoded: numpy array with all numeric values
            encoders: dict mapping column name to {value: code} mapping
        """
        Z_encoded = np.zeros((len(Z_df), len(Z_df.columns)), dtype=float)
        encoders: Dict[str, Dict[object, int]] = {}
        
        for j, col in enumerate(Z_df.columns):
            if col in self.cat_vars:
                # Encode categorical column
                series = Z_df[col].astype(str)
                unique_vals = series.unique()
                code_map = {v: i for i, v in enumerate(unique_vals)}
                encoders[col] = code_map
                Z_encoded[:, j] = series.map(code_map).values.astype(float)
            else:
                # Continuous column - convert to numeric
                Z_encoded[:, j] = pd.to_numeric(Z_df[col], errors="coerce").values.astype(float)
        
        return Z_encoded, encoders

    # -------------------------
    # Training: continuous
    # -------------------------
    def _train_continuous_feature(
        self,
        f: str,
        X_current: pd.DataFrame,
        X_missing: pd.DataFrame,
        mask_df: pd.DataFrame,
        P_f: np.ndarray,
        alpha: float,
    ) -> Tuple[nn.Module, np.ndarray]:
        """
        Train regression CPFA for continuous target.
        """
        device = self.device
        set_global_seed(self.cfg.seed + (hash(f) % 10000))

        # Prepare data
        y_full = pd.to_numeric(X_current[f], errors="coerce").values.astype(float)
        Z_df = X_current.drop(columns=[f])
        
        # FIXED: Properly encode categorical columns instead of coercing to NaN
        Z_full, _ = self._encode_dataframe_for_training(Z_df)

        mask_orig_miss = mask_df[f].values
        mask_present = ~mask_orig_miss

        # Missingness mask for SNI-M
        mask_aware = (self.cfg.variant == "SNI-M")
        if mask_aware:
            M_full = (~mask_df.drop(columns=[f]).values).astype(float)
        else:
            M_full = None

        # Standardize
        scaler_Z = StandardScaler()
        scaler_y = StandardScaler()

        Z_obs = Z_full[mask_present]
        y_obs = y_full[mask_present]

        # Handle NaN in Z_obs (shouldn't happen now, but be safe)
        Z_obs = np.nan_to_num(Z_obs, nan=0.0)
        Z_obs_scaled = scaler_Z.fit_transform(Z_obs)
        y_obs_scaled = scaler_y.fit_transform(y_obs.reshape(-1, 1)).flatten()

        Z_full_scaled = scaler_Z.transform(np.nan_to_num(Z_full, nan=0.0))
        y_full_scaled = np.zeros_like(y_full, dtype=float)
        y_full_scaled[mask_present] = y_obs_scaled

        # Train/val split
        idx_tr, idx_va = self._split_indices(y_full_scaled, mask_present, test_size=0.2, is_classification=False)

        # Pseudo-mask
        M_tr = np.random.rand(len(idx_tr)) < self.cfg.mask_fraction
        train_used_idx = idx_tr[~M_tr]
        val_idx = idx_va

        # Tensors
        X_train = torch.tensor(Z_full_scaled[train_used_idx], dtype=torch.float32)
        y_train = torch.tensor(y_full_scaled[train_used_idx], dtype=torch.float32).unsqueeze(1)
        if mask_aware:
            M_train = torch.tensor(M_full[train_used_idx], dtype=torch.float32)
            train_ds = TensorDataset(X_train, M_train, y_train)
        else:
            train_ds = TensorDataset(X_train, y_train)

        X_val = torch.tensor(Z_full_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_full_scaled[val_idx], dtype=torch.float32).unsqueeze(1)
        if mask_aware:
            M_val = torch.tensor(M_full[val_idx], dtype=torch.float32)

        # Model
        cat_indices = self._get_categorical_indices_excluding_target(f, self.all_vars)

        model = EnhancedCPFA(
            input_dim=Z_full.shape[1],
            emb_dim=self.cfg.emb_dim,
            num_heads=self.cfg.num_heads,
            hidden_dims=list(self.cfg.hidden_dims),
            output_dim=1,
            is_classification=False,
            cat_indices=cat_indices,
            use_cat_embedding=self.cfg.use_cat_embedding,
            use_multiscale=False,
            mask_aware=mask_aware,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=1e-6)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.cfg.use_gpu,
        )

        best_val = float("inf")
        best_state = None
        P_t = torch.tensor(P_f, dtype=torch.float32, device=device)
        patience_counter = 0

        self.logs.setdefault(f, {"tr": [], "va": []})
        self.lambda_trace_per_head.setdefault(f, [])

        for epoch in range(self.cfg.epochs):
            model.train()
            total_loss = 0.0
            lambdas_epoch = []

            for batch in train_loader:
                if mask_aware:
                    Xb, Mb, yb = batch
                    Xb = Xb.to(device, non_blocking=True)
                    Mb = Mb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    y_hat, A_heads, _, lambdas = model(Xb, M=Mb)
                else:
                    Xb, yb = batch
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    y_hat, A_heads, _, lambdas = model(Xb)

                loss_recon = criterion(y_hat, yb)

                # v0.3: Support lambda_mode='fixed'
                if self.cfg.lambda_mode == "fixed":
                    lambdas_eff = torch.full_like(lambdas, float(self.cfg.lambda_fixed_value))
                else:
                    lambdas_eff = lambdas

                if self.cfg.variant == "HardPrior":
                    lambdas_eff = torch.full_like(lambdas, float(self.cfg.hard_prior_lambda))
                    A_vec_heads = A_heads.mean(dim=1)
                    loss_prior = model.compute_prior_loss(A_vec_heads, P_t, lambdas_eff, alpha)
                elif self.cfg.variant == "NoPrior":
                    loss_prior = torch.tensor(0.0, device=device)
                else:
                    A_vec_heads = A_heads.mean(dim=1)
                    loss_prior = model.compute_prior_loss(A_vec_heads, P_t, lambdas_eff, alpha)

                loss = loss_recon + loss_prior

                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item()) * Xb.size(0)
                lambdas_epoch.append(lambdas.detach().cpu().numpy().tolist())

            avg_train = total_loss / max(1, len(train_loader.dataset))
            self.logs[f]["tr"].append(avg_train)

            # Validation
            model.eval()
            with torch.no_grad():
                if len(val_idx) > 0:
                    Xv = X_val.to(device)
                    yv = y_val.to(device)
                    if mask_aware:
                        Mv = M_val.to(device)
                        y_hat_v, A_heads_v, _, lambdas_v = model(Xv, M=Mv)
                    else:
                        y_hat_v, A_heads_v, _, lambdas_v = model(Xv)
                    loss_recon_v = criterion(y_hat_v, yv)

                    # v0.3: Support lambda_mode='fixed'
                    if self.cfg.lambda_mode == "fixed":
                        lambdas_v_eff = torch.full_like(lambdas_v, float(self.cfg.lambda_fixed_value))
                    else:
                        lambdas_v_eff = lambdas_v

                    if self.cfg.variant == "HardPrior":
                        lambdas_v_eff = torch.full_like(lambdas_v, float(self.cfg.hard_prior_lambda))
                        A_vec_heads_v = A_heads_v.mean(dim=1)
                        loss_prior_v = model.compute_prior_loss(A_vec_heads_v, P_t, lambdas_v_eff, alpha)
                    elif self.cfg.variant == "NoPrior":
                        loss_prior_v = torch.tensor(0.0, device=device)
                    else:
                        A_vec_heads_v = A_heads_v.mean(dim=1)
                        loss_prior_v = model.compute_prior_loss(A_vec_heads_v, P_t, lambdas_v_eff, alpha)

                    val_loss = float((loss_recon_v + loss_prior_v).item())
                else:
                    val_loss = float("inf")
            self.logs[f]["va"].append(val_loss)

            # Lambda trace
            if len(lambdas_epoch) > 0:
                avg_l = np.mean(np.asarray(lambdas_epoch), axis=0).tolist()
            else:
                avg_l = [0.0] * self.cfg.num_heads
            self.lambda_trace_per_head[f].append(avg_l)

            scheduler.step()

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience and epoch > self.cfg.min_epochs:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Attention map
        A_avg_mat_all = self._compute_global_attention_map(model, Z_full_scaled, M_full if mask_aware else None, device)
        self.attention_maps[f] = A_avg_mat_all

        # Predict missing
        missing_idx = np.where(mask_orig_miss)[0]
        if len(missing_idx) > 0:
            X_miss = torch.tensor(Z_full_scaled[missing_idx], dtype=torch.float32).to(device)
            if mask_aware:
                M_miss = torch.tensor(M_full[missing_idx], dtype=torch.float32).to(device)
                with torch.no_grad():
                    y_hat_miss, _, _, _ = model(X_miss, M=M_miss)
            else:
                with torch.no_grad():
                    y_hat_miss, _, _, _ = model(X_miss)
            y_pred_scaled = y_hat_miss.cpu().numpy().flatten()
            preds = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            preds = np.array([], dtype=float)

        return model, preds

    # -------------------------
    # Training: categorical
    # -------------------------
    def _train_categorical_feature(
        self,
        f: str,
        X_current: pd.DataFrame,
        X_missing: pd.DataFrame,
        mask_df: pd.DataFrame,
        P_f: np.ndarray,
        alpha: float,
    ) -> Tuple[nn.Module, np.ndarray]:
        """
        Train classification CPFA for categorical target.
        """
        device = self.device
        set_global_seed(self.cfg.seed + (hash(f) % 10000))

        le = self.encoders[f]
        y_str = X_current[f].astype(str).values
        observed_classes = np.unique(y_str[~mask_df[f].values])
        le.fit(observed_classes)
        n_classes = len(le.classes_)

        y_full = np.zeros(len(y_str), dtype=int)
        for i, val in enumerate(y_str):
            if val in le.classes_:
                y_full[i] = int(le.transform([val])[0])
            else:
                y_full[i] = 0

        Z_df = X_current.drop(columns=[f])
        
        # FIXED: Properly encode categorical columns instead of coercing to NaN
        Z_full, _ = self._encode_dataframe_for_training(Z_df)

        mask_orig_miss = mask_df[f].values
        mask_present = ~mask_orig_miss

        mask_aware = (self.cfg.variant == "SNI-M")
        if mask_aware:
            M_full = (~mask_df.drop(columns=[f]).values).astype(float)
        else:
            M_full = None

        # Standardize Z
        scaler_Z = StandardScaler()
        Z_obs = Z_full[mask_present]
        
        # Handle NaN in Z_obs (shouldn't happen now, but be safe)
        Z_obs = np.nan_to_num(Z_obs, nan=0.0)
        Z_obs_scaled = scaler_Z.fit_transform(Z_obs)
        Z_full_scaled = scaler_Z.transform(np.nan_to_num(Z_full, nan=0.0))

        # Train/val split
        idx_tr, idx_va = self._split_indices(y_full, mask_present, test_size=0.2, is_classification=True)

        # Pseudo-mask (stratified)
        y_tr = y_full[idx_tr]
        pseudo_mask = categorical_aware_pseudo_masking(y_tr, self.cfg.mask_fraction)
        train_used_idx = idx_tr[~pseudo_mask]
        val_idx = idx_va

        # Tensors
        X_train = torch.tensor(Z_full_scaled[train_used_idx], dtype=torch.float32)
        y_train = torch.tensor(y_full[train_used_idx], dtype=torch.long)
        if mask_aware:
            M_train = torch.tensor(M_full[train_used_idx], dtype=torch.float32)
            train_ds = TensorDataset(X_train, M_train, y_train)
        else:
            train_ds = TensorDataset(X_train, y_train)

        X_val = torch.tensor(Z_full_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_full[val_idx], dtype=torch.long)
        if mask_aware:
            M_val = torch.tensor(M_full[val_idx], dtype=torch.float32)

        # Model
        cat_indices = self._get_categorical_indices_excluding_target(f, self.all_vars)

        if self.cfg.use_dual_path:
            model = DualPathCPFA(
                input_dim=Z_full.shape[1],
                emb_dim=self.cfg.emb_dim,
                num_heads=self.cfg.num_heads,
                hidden_dims=list(self.cfg.hidden_dims),
                output_dim=n_classes,
                is_classification=True,
                cat_indices=cat_indices,
                use_cat_embedding=self.cfg.use_cat_embedding,
                use_multiscale=self.cfg.use_multiscale,
                mask_aware=mask_aware,
            ).to(device)
        else:
            model = EnhancedCPFA(
                input_dim=Z_full.shape[1],
                emb_dim=self.cfg.emb_dim,
                num_heads=self.cfg.num_heads,
                hidden_dims=list(self.cfg.hidden_dims),
                output_dim=n_classes,
                is_classification=True,
                cat_indices=cat_indices,
                use_cat_embedding=self.cfg.use_cat_embedding,
                use_multiscale=self.cfg.use_multiscale,
                mask_aware=mask_aware,
            ).to(device)

        # Class weights
        y_train_np = y_train.numpy()
        present = np.unique(y_train_np)
        if len(present) < n_classes:
            cw_present = compute_class_weight("balanced", classes=present, y=y_train_np)
            class_weights = np.ones(n_classes, dtype=float)
            class_weights[present.astype(int)] = cw_present.astype(float)
        else:
            class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train_np)

        # v0.3: Class-balanced weighting for focal loss
        if self.cfg.cat_balance_mode != "none":
            counts = np.bincount(y_train_np, minlength=n_classes).astype(float)
            counts = np.maximum(counts, 1.0)  # avoid division by zero
            freqs = counts / counts.sum()
            if self.cfg.cat_balance_mode == "inverse_freq":
                balance_w = 1.0 / freqs
            elif self.cfg.cat_balance_mode == "sqrt_inverse_freq":
                balance_w = 1.0 / np.sqrt(freqs)
            else:
                balance_w = np.ones(n_classes, dtype=float)
            # Normalize so mean = 1, then cap at 10x
            balance_w = balance_w / balance_w.mean()
            balance_w = np.minimum(balance_w, 10.0)
            class_weights = balance_w

        criterion = self.cat_trainer.create_loss_function(class_weights, n_classes, self.cfg.label_smoothing_epsilon)

        # v0.3: Categorical-specific LR multiplier
        cat_lr = self.cfg.lr * self.cfg.cat_lr_mult
        if self.cfg.cat_lr_mult != 1.0:
            # Split parameters: MLP head gets multiplied LR, rest gets base LR
            head_params = list(model.mlp.parameters())
            head_ids = {id(p) for p in head_params}
            base_params = [p for p in model.parameters() if id(p) not in head_ids]
            optimizer = torch.optim.AdamW([
                {"params": base_params, "lr": self.cfg.lr},
                {"params": head_params, "lr": cat_lr},
            ], weight_decay=self.cfg.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=1e-6)

        # Balanced sampler
        cls_counts = np.bincount(y_train.numpy(), minlength=n_classes)
        cls_w = 1.0 / (cls_counts + 1e-6)
        sample_w = cls_w[y_train.numpy()]
        sample_w_t = torch.tensor(sample_w, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_w_t, num_samples=len(sample_w_t), replacement=True)

        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg.batch_size, 
            sampler=sampler, 
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.cfg.use_gpu,
        )

        best_acc = 0.0
        best_state = None
        P_t = torch.tensor(P_f, dtype=torch.float32, device=device)
        patience_counter = 0

        self.logs.setdefault(f, {"tr": [], "va": []})
        self.lambda_trace_per_head.setdefault(f, [])

        for epoch in range(self.cfg.epochs):
            prior_mode = "learned"
            if self.cfg.variant == "NoPrior":
                prior_mode = "none"
            elif self.cfg.variant == "HardPrior":
                prior_mode = "hard"

            avg_train_loss = self.cat_trainer.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                P_f_tensor=P_t,
                alpha_g_minus_1=alpha,
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                device=device,
                mask_aware=mask_aware,
                prior_mode=prior_mode,
                hard_prior_lambda=self.cfg.hard_prior_lambda,
                lambda_mode=self.cfg.lambda_mode,
                lambda_fixed_value=self.cfg.lambda_fixed_value,
            )
            self.logs[f]["tr"].append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                if len(val_idx) > 0:
                    Xv = X_val.to(device)
                    yv = y_val.to(device)
                    if mask_aware:
                        Mv = M_val.to(device)
                        logits, A_heads_v, _, lambdas_v = model(Xv, M=Mv)
                    else:
                        logits, A_heads_v, _, lambdas_v = model(Xv)

                    preds = logits.argmax(dim=1)
                    acc = float((preds == yv).float().mean().item())

                    loss_recon_v = criterion(logits, yv)

                    # v0.3: Support lambda_mode='fixed'
                    if self.cfg.lambda_mode == "fixed":
                        lambdas_v_eff = torch.full_like(lambdas_v, float(self.cfg.lambda_fixed_value))
                    else:
                        lambdas_v_eff = lambdas_v

                    if self.cfg.variant == "HardPrior":
                        lambdas_v_eff = torch.full_like(lambdas_v, float(self.cfg.hard_prior_lambda))
                        A_vec_heads_v = A_heads_v.mean(dim=1)
                        loss_prior_v = model.compute_prior_loss(A_vec_heads_v, P_t, lambdas_v_eff, alpha)
                    elif self.cfg.variant == "NoPrior":
                        loss_prior_v = torch.tensor(0.0, device=device)
                    else:
                        A_vec_heads_v = A_heads_v.mean(dim=1)
                        loss_prior_v = model.compute_prior_loss(A_vec_heads_v, P_t, lambdas_v_eff, alpha)

                    val_loss = float((loss_recon_v + loss_prior_v).item())
                else:
                    acc = 0.0
                    val_loss = float("inf")
                    lambdas_v = torch.zeros(self.cfg.num_heads, device=device)

            self.logs[f]["va"].append(val_loss)
            self.lambda_trace_per_head[f].append(lambdas_v.detach().cpu().numpy().tolist())

            scheduler.step()

            # Early stopping based on accuracy
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience and epoch > self.cfg.min_epochs:
                    break

            # Very good accuracy - can stop early
            if epoch > self.cfg.min_epochs and acc > 0.98:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Attention map
        A_avg_mat_all = self._compute_global_attention_map(model, Z_full_scaled, M_full if mask_aware else None, device)
        self.attention_maps[f] = A_avg_mat_all

        # Predict missing
        missing_idx = np.where(mask_orig_miss)[0]
        if len(missing_idx) > 0:
            X_miss = torch.tensor(Z_full_scaled[missing_idx], dtype=torch.float32).to(device)
            if mask_aware:
                M_miss = torch.tensor(M_full[missing_idx], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits, _, _, _ = model(X_miss, M=M_miss)
            else:
                with torch.no_grad():
                    logits, _, _, _ = model(X_miss)
            pred_labels = logits.argmax(dim=1).cpu().numpy()
            preds = le.inverse_transform(pred_labels)
        else:
            preds = np.array([], dtype=object)

        return model, preds

    # -------------------------
    # Helpers
    # -------------------------
    def _split_indices(self, y_full: np.ndarray, mask_present: np.ndarray, test_size: float, is_classification: bool):
        idx_obs = np.where(mask_present)[0]
        y_obs = y_full[idx_obs]
        rs = self.cfg.seed

        if is_classification:
            uniq, counts = np.unique(y_obs, return_counts=True)
            if len(uniq) < 2 or counts.min() < 2:
                idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs)
            else:
                idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs, stratify=y_obs)
            return np.asarray(idx_tr, dtype=int), np.asarray(idx_va, dtype=int)

        # Regression: stratify by quantile bins
        try:
            y_num = np.asarray(y_obs, dtype=float)
            q = np.quantile(y_num, np.linspace(0, 1, 11))
            q = np.unique(q)
            if len(q) >= 3:
                bins = np.digitize(y_num, q[1:-1], right=True)
                if np.min(np.bincount(bins)) >= 2:
                    idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs, stratify=bins)
                else:
                    idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs)
            else:
                idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs)
        except Exception:
            idx_tr, idx_va = train_test_split(idx_obs, test_size=test_size, random_state=rs)

        return np.asarray(idx_tr, dtype=int), np.asarray(idx_va, dtype=int)

    def _compute_global_attention_map(
        self,
        model: nn.Module,
        Z_full: np.ndarray,
        M_full: Optional[np.ndarray],
        device: torch.device,
    ) -> np.ndarray:
        """
        Compute attention matrix averaged over all samples.
        """
        model.eval()
        n = Z_full.shape[0]
        bs = max(64, min(512, self.cfg.batch_size))
        total = 0
        A_sum = None

        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(n, start + bs)
                Zb = torch.tensor(Z_full[start:end], dtype=torch.float32).to(device)
                if M_full is not None:
                    Mb = torch.tensor(M_full[start:end], dtype=torch.float32).to(device)
                    _, A_heads, _, _ = model(Zb, M=Mb)
                else:
                    _, A_heads, _, _ = model(Zb)
                batch_size = end - start
                A_batch = A_heads.mean(dim=0).detach().cpu().numpy()
                if A_sum is None:
                    A_sum = A_batch * batch_size
                else:
                    A_sum += A_batch * batch_size
                total += batch_size

        A_avg = A_sum / max(1, total)
        return A_avg