# TabCSDI_v1.py
# -*- coding: utf-8 -*-
"""
TabCSDI - Conditional Score-based Diffusion for Tabular Imputation (v1)

Self-contained simplified implementation of CSDI adapted for tabular mixed-type
data. The core architecture follows the CSDI paper, with adaptations from the
TabCSDI work for feature tokenization on mixed continuous/categorical columns.

Key design:
- DDPM forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
- Reverse process: learned denoiser predicts noise epsilon
- Feature tokenization: per-column linear embedding + positional encoding
- Transformer-based denoiser with conditioning on observed values
- Continuous features: standardize -> diffuse -> denoise -> destandardize
- Categorical features: one-hot encode -> diffuse -> denoise -> argmax decode
- Multiple sampling (n_samples) with median aggregation for point imputation

Reference:
    Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021).
    CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series
    Imputation. NeurIPS 2021.

    Zheng, A. & Charoenphakdee, N. (2022).
    Diffusion models for missing value imputation in tabular data.
    NeurIPS 2022 Workshop (TabCSDI adaptation).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Diffusion schedule utilities
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule (Nichol & Dhariwal 2021). More gentle than linear."""
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_bar = f / f[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return np.clip(betas, 1e-5, 0.999).astype(np.float32)


class DiffusionSchedule:
    """Pre-computed DDPM noise schedule tensors."""

    def __init__(self, T: int, device: torch.device):
        betas = _cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)

        self.T = T
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
        self.alphas_bar = torch.tensor(alphas_bar, dtype=torch.float32, device=device)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # Posterior variance for reverse step
        alphas_bar_prev = np.concatenate([[1.0], alphas_bar[:-1]])
        self.posterior_var = torch.tensor(
            betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
            dtype=torch.float32,
            device=device,
        )


# ---------------------------------------------------------------------------
# Transformer denoiser
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal encoding for diffusion timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FeatureTokenizer(nn.Module):
    """Per-column linear embedding for tabular features."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> tokens: (B, D, d_model)
        tokens = []
        for i in range(self.n_features):
            tokens.append(self.embeddings[i](x[:, i : i + 1]))
        return torch.stack(tokens, dim=1)


class DenoisingTransformer(nn.Module):
    """
    Transformer-based epsilon predictor for tabular diffusion.

    Architecture:
    - Feature tokenizer: per-column linear embedding
    - Conditioning: observed features embedded + missingness mask
    - Timestep: sinusoidal encoding projected and added
    - Transformer encoder layers for cross-feature interaction
    - Per-feature output heads predict noise epsilon
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Noisy input tokenizer
        self.input_tokenizer = FeatureTokenizer(n_features, d_model)
        # Condition (observed values) tokenizer
        self.cond_tokenizer = FeatureTokenizer(n_features, d_model)

        # Mask embedding (1-dim → d_model per feature)
        self.mask_embedding = nn.Linear(1, d_model)

        # Timestep embedding
        self.time_enc = SinusoidalPositionEncoding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)

        # Learnable feature-position embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Fusion: combine noisy input + condition + mask
        self.fusion = nn.Linear(d_model * 3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-feature output projection → predict noise for each feature
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(n_features)]
        )

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, D) noisy features
        x_cond: torch.Tensor,    # (B, D) observed features (0 at missing)
        mask: torch.Tensor,       # (B, D) 1=observed, 0=missing
        t: torch.Tensor,          # (B,) timestep indices
    ) -> torch.Tensor:
        B = x_noisy.size(0)

        # Tokenize
        tok_noisy = self.input_tokenizer(x_noisy)       # (B, D, d)
        tok_cond = self.cond_tokenizer(x_cond)           # (B, D, d)
        tok_mask = self.mask_embedding(mask.unsqueeze(-1))  # (B, D, d)

        # Fuse
        fused = self.fusion(torch.cat([tok_noisy, tok_cond, tok_mask], dim=-1))  # (B, D, d)

        # Add positional encoding
        fused = fused + self.pos_emb

        # Add timestep encoding (broadcast to all feature tokens)
        t_emb = self.time_proj(self.time_enc(t))  # (B, d)
        fused = fused + t_emb.unsqueeze(1)

        # Transformer
        out = self.transformer(fused)  # (B, D, d)

        # Per-feature output
        eps_parts = []
        for i in range(self.n_features):
            eps_parts.append(self.output_heads[i](out[:, i, :]))
        eps_pred = torch.cat(eps_parts, dim=-1)  # (B, D)

        return eps_pred


# ---------------------------------------------------------------------------
# TabCSDI Imputer
# ---------------------------------------------------------------------------

class TabCSDIImputer:
    """
    TabCSDI: Conditional Score-based Diffusion Imputer for tabular data.

    Training:
    - Encode data: standardize continuous, one-hot encode categorical
    - For each batch: sample t ~ Uniform(1..T), sample noise, compute x_t
    - Train denoiser to predict noise (only at originally-missing positions
      are imputed; observed positions provide conditioning)

    Imputation:
    - Start from x_T ~ N(0, I) at missing positions
    - Reverse diffuse T steps conditioned on observed values
    - Repeat n_samples times, aggregate via median
    - Decode: destandardize continuous, argmax for categorical
    """

    def __init__(
        self,
        categorical_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        seed: int = 42,
        use_gpu: bool = False,
        # Diffusion
        diffusion_steps: int = 50,
        n_samples: int = 10,
        # Architecture
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        # Training
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        min_epochs: int = 50,
        early_stopping_patience: int = 20,
    ):
        self.categorical_vars = list(categorical_vars or [])
        self.continuous_vars = list(continuous_vars or [])
        self.seed = seed
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.diffusion_steps = diffusion_steps
        self.n_samples = n_samples
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_epochs = min_epochs
        self.early_stopping_patience = early_stopping_patience

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.models_: Dict = {}

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _encode(
        self, df: pd.DataFrame, X_complete: pd.DataFrame, fit: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode mixed-type DataFrame to numeric matrix.

        Returns:
            data: (N, D_enc) float array  (NaN preserved)
            mask: (N, D_enc) float array  (1=observed, 0=missing)
        """
        blocks: List[np.ndarray] = []
        mask_blocks: List[np.ndarray] = []

        if fit:
            self._col_order: List[str] = []
            self._cont_means: Dict[str, float] = {}
            self._cont_stds: Dict[str, float] = {}
            self._cat_categories: Dict[str, list] = {}
            self._enc_slices: Dict[str, Tuple[int, int, str]] = {}  # col -> (start, end, type)

        idx = 0
        for col in self.continuous_vars:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
            obs_mask = (~np.isnan(vals)).astype(np.float32)

            if fit:
                # R0: standardization constants came from the GROUND-TRUTH table.
                # R1 (X_complete is None): from the observed cells of df itself.
                ref_vals = vals if X_complete is None else pd.to_numeric(
                    X_complete[col], errors="coerce"
                ).values
                self._cont_means[col] = float(np.nanmean(ref_vals))
                self._cont_stds[col] = float(np.nanstd(ref_vals))
                if self._cont_stds[col] < 1e-8:
                    self._cont_stds[col] = 1.0
                self._col_order.append(col)
                self._enc_slices[col] = (idx, idx + 1, "cont")

            standardized = (vals - self._cont_means[col]) / self._cont_stds[col]
            standardized = np.where(np.isnan(standardized), 0.0, standardized)

            blocks.append(standardized.reshape(-1, 1))
            mask_blocks.append(obs_mask.reshape(-1, 1))
            idx += 1

        for col in self.categorical_vars:
            if col not in df.columns:
                continue

            if fit:
                # R0: full label set from the GROUND-TRUTH table.
                # R1 (X_complete is None): observed labels of df only.
                src = df[col] if X_complete is None else X_complete[col]
                cats = src.dropna().unique().tolist()
                try:
                    cats = sorted(cats)
                except Exception:
                    pass
                self._cat_categories[col] = cats
                self._col_order.append(col)
                n_cat = len(cats)
                self._enc_slices[col] = (idx, idx + n_cat, "cat")
            else:
                cats = self._cat_categories[col]
                n_cat = len(cats)

            cat_map = {v: i for i, v in enumerate(cats)}
            n = len(df)
            ohe = np.zeros((n, n_cat), dtype=np.float32)
            obs_mask = np.zeros((n, n_cat), dtype=np.float32)

            for i, val in enumerate(df[col]):
                if pd.isna(val):
                    # Missing: leave zeros, mask=0
                    pass
                elif val in cat_map:
                    ohe[i, cat_map[val]] = 1.0
                    obs_mask[i, :] = 1.0
                else:
                    # Unknown category: treat as observed mode 0
                    ohe[i, 0] = 1.0
                    obs_mask[i, :] = 1.0

            blocks.append(ohe)
            mask_blocks.append(obs_mask)
            idx += n_cat

        if fit:
            self._enc_dim = idx

        data = np.hstack(blocks).astype(np.float32)
        mask = np.hstack(mask_blocks).astype(np.float32)
        return data, mask

    def _decode(self, data: np.ndarray) -> pd.DataFrame:
        """Decode encoded numeric matrix back to mixed-type DataFrame."""
        n = data.shape[0]
        result: Dict[str, np.ndarray] = {}

        for col in self._col_order:
            start, end, ctype = self._enc_slices[col]
            if ctype == "cont":
                vals = data[:, start] * self._cont_stds[col] + self._cont_means[col]
                result[col] = vals
            else:
                logits = data[:, start:end]
                indices = np.argmax(logits, axis=1)
                cats = self._cat_categories[col]
                result[col] = np.array([cats[min(i, len(cats) - 1)] for i in indices])

        return pd.DataFrame(result, columns=self._col_order)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, data: np.ndarray, mask: np.ndarray) -> nn.Module:
        """Train the denoiser on observed data with self-supervised masking."""
        device = self.device
        N, D = data.shape

        schedule = DiffusionSchedule(self.diffusion_steps, device)

        model = DenoisingTransformer(
            n_features=D,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        # Training uses self-supervised masking:
        # Among observed positions, randomly mask some and train to denoise them.
        # This teaches the model to impute from partial observations.
        data_t = torch.tensor(data, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

        dataset = TensorDataset(data_t, mask_t)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(dataset) > self.batch_size,
        )

        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for x_batch, m_batch in loader:
                B = x_batch.size(0)

                # Self-supervised masking: randomly hide ~30% of observed positions
                train_mask = m_batch.clone()
                hide_prob = torch.rand_like(m_batch)
                hide = (hide_prob < 0.3) & (m_batch > 0.5)
                train_mask[hide] = 0.0

                # Condition = observed (not hidden), target = everything observed
                x_cond = x_batch * train_mask

                # Sample timestep
                t = torch.randint(0, self.diffusion_steps, (B,), device=device)

                # Forward diffusion on full x_batch
                noise = torch.randn_like(x_batch)
                sqrt_ab = schedule.sqrt_alphas_bar[t].unsqueeze(-1)
                sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t].unsqueeze(-1)
                x_noisy = sqrt_ab * x_batch + sqrt_1mab * noise

                # Replace observed (not hidden) positions with clean values
                # This is the "conditional" part: observed positions keep their values
                x_noisy = x_noisy * (1 - train_mask) + x_batch * train_mask

                # Predict noise
                eps_pred = model(x_noisy, x_cond, train_mask, t)

                # Loss: only on hidden positions (self-supervised target)
                loss_mask = hide.float()  # positions that were hidden
                n_loss_elems = loss_mask.sum() + 1e-8
                loss = ((eps_pred - noise) ** 2 * loss_mask).sum() / n_loss_elems

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * B
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, N)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.early_stopping_patience and epoch >= self.min_epochs:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Reverse diffusion sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_once(
        self,
        model: nn.Module,
        data: np.ndarray,
        mask: np.ndarray,
        schedule: DiffusionSchedule,
    ) -> np.ndarray:
        """Run one full reverse diffusion pass to impute missing values."""
        device = self.device
        N, D = data.shape

        data_t = torch.tensor(data, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        x_cond = data_t * mask_t

        # Initialize missing positions with pure noise
        x = torch.randn(N, D, device=device)
        # Keep observed positions
        x = x * (1 - mask_t) + data_t * mask_t

        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t_val in reversed(range(schedule.T)):
            t_tensor = torch.full((N,), t_val, dtype=torch.long, device=device)

            # Predict noise (process in mini-batches for memory)
            eps_pred = self._batch_predict(model, x, x_cond, mask_t, t_tensor)

            # DDPM reverse step (only at missing positions)
            alpha_t = schedule.alphas[t_val]
            alpha_bar_t = schedule.alphas_bar[t_val]
            beta_t = schedule.betas[t_val]

            # Mean of reverse distribution
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_pred)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(schedule.posterior_var[t_val])
                x_new = mean + sigma * noise
            else:
                x_new = mean

            # Replace: keep observed values, update only missing
            x = x_new * (1 - mask_t) + data_t * mask_t

        return x.cpu().numpy()

    def _batch_predict(
        self,
        model: nn.Module,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict in mini-batches to avoid OOM on large datasets."""
        N = x.size(0)
        bs = max(256, self.batch_size * 2)
        if N <= bs:
            return model(x, x_cond, mask, t)

        parts = []
        for start in range(0, N, bs):
            end = min(N, start + bs)
            part = model(
                x[start:end],
                x_cond[start:end],
                mask[start:end],
                t[start:end],
            )
            parts.append(part)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def impute(
        self,
        X_complete,
        X_missing: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Impute missing values using conditional score-based diffusion.

        Args:
            X_complete: Ground-truth DataFrame, or ``None`` for the R1 de-leaked
                path. R0 used it for two things and both were real leaks:
                per-column mean/std for standardization (``:310-312``) and the
                full categorical label set (``:329-330``). With ``None`` both are
                derived from the observed cells of ``X_missing``. The diffusion
                schedule, the denoiser and the sampling loop are unchanged.
            X_missing: DataFrame with NaN at missing positions.

        Returns:
            X_imputed: Imputed DataFrame (same shape/columns as X_missing).
            models_: Dict containing the trained denoiser.
        """
        # Encode
        data, mask = self._encode(X_missing, X_complete, fit=True)

        # Train denoiser
        model = self._train(data, mask)
        self.models_["denoiser"] = model

        # Sample multiple times and aggregate
        schedule = DiffusionSchedule(self.diffusion_steps, self.device)

        samples: List[np.ndarray] = []
        for s in range(self.n_samples):
            # Re-seed per sample for diversity but reproducibility
            torch.manual_seed(self.seed + s + 1)
            sample = self._sample_once(model, data, mask, schedule)
            samples.append(sample)

        # Aggregate: median over samples
        stacked = np.stack(samples, axis=0)  # (n_samples, N, D)
        imputed_enc = np.median(stacked, axis=0)  # (N, D)

        # For observed positions, keep original encoded values
        imputed_enc = np.where(mask > 0.5, data, imputed_enc)

        # Decode
        df_imputed = self._decode(imputed_enc)

        # Restore original column order from X_missing
        all_cols = [c for c in X_missing.columns if c in df_imputed.columns]
        df_imputed = df_imputed[all_cols]
        df_imputed.index = X_missing.index

        return df_imputed, self.models_


# Backward-compatible alias
tabCSDIImputer = TabCSDIImputer


def _run_self_test() -> None:
    """Quick synthetic data self-test (no CLI args)."""
    print("TabCSDI v1 self-test")
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

    imputer = TabCSDIImputer(
        categorical_vars=["cat1", "cat2"],
        continuous_vars=["x1", "x2", "x3"],
        seed=42,
        diffusion_steps=20,   # Fewer steps for quick test
        n_samples=5,
        epochs=30,            # Fewer epochs for quick test
        d_model=64,
        n_layers=2,
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
            std_true = np.std(true_vals)
            nrmse_val = rmse / std_true if std_true > 0 else rmse
            print(f"  {col}: NRMSE = {nrmse_val:.4f}")

    for col in ["cat1", "cat2"]:
        mask = df_missing[col].isna()
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values.astype(str)
            imp_vals = df_imputed.loc[mask, col].values.astype(str)
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
        description="TabCSDI CLI — impute missing data and write metrics_summary.json"
    )
    ap.add_argument("--input-complete", required=True, help="Path to complete CSV")
    ap.add_argument("--input-missing", required=True, help="Path to missing CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--categorical-vars", type=str, default="", help="Comma-separated categorical column names")
    ap.add_argument("--continuous-vars", type=str, default="", help="Comma-separated continuous column names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--diffusion-steps", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
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

    print(f"[TabCSDI CLI] complete shape: {X_complete.shape}, missing shape: {X_missing.shape}")
    print(f"[TabCSDI CLI] categorical_vars={cat_vars}, continuous_vars={cont_vars}")
    print(f"[TabCSDI CLI] use_gpu={args.use_gpu}, diffusion_steps={args.diffusion_steps}, n_samples={args.n_samples}")

    t0 = time.time()

    imputer = TabCSDIImputer(
        categorical_vars=cat_vars,
        continuous_vars=cont_vars,
        seed=args.seed,
        use_gpu=args.use_gpu,
        diffusion_steps=args.diffusion_steps,
        n_samples=args.n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
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
        "method": "TabCSDI",
        "seed": args.seed,
        "use_gpu": args.use_gpu,
        "runtime_sec": round(runtime_sec, 3),
    })

    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[TabCSDI CLI] Done in {runtime_sec:.1f}s → {outdir / 'metrics_summary.json'}")
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
