# cpfa.py - Controllable-Prior Feature Attention (CPFA) Module
# SNI v0.3 - lambda ablation support added

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import FocalCrossEntropyLoss, LabelSmoothingCrossEntropy


class CategoricalEmbeddingModule(nn.Module):
    """
    Enhanced embedding for categorical features with cross-category attention.
    Fuses categorical embeddings with continuous features for richer representations.
    """
    def __init__(self, input_dim: int, cat_indices: List[int], emb_dim: int = 32, shared_emb_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cat_indices = list(cat_indices)
        self.cont_indices = [i for i in range(input_dim) if i not in self.cat_indices]
        self.shared_emb_dim = int(shared_emb_dim)

        # Categorical feature embeddings
        self.cat_embeddings = nn.ModuleList([nn.Linear(1, emb_dim) for _ in self.cat_indices])

        # Continuous feature embedding
        self.cont_embedding = nn.Linear(len(self.cont_indices), shared_emb_dim) if len(self.cont_indices) > 0 else None

        # Cross-attention for categorical tokens
        self.cross_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, dropout=0.1, batch_first=True)

        # Output projection
        output_dim = emb_dim * len(self.cat_indices) + (shared_emb_dim if self.cont_embedding is not None else 0)
        self.output_proj = nn.Linear(output_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Categorical tokens
        cat_tokens = []
        for i, idx in enumerate(self.cat_indices):
            cat_val = x[:, idx:idx + 1]
            cat_emb = self.cat_embeddings[i](cat_val)
            cat_tokens.append(cat_emb.unsqueeze(1))

        if len(cat_tokens) > 0:
            cat_stack = torch.cat(cat_tokens, dim=1)
            cat_attended, _ = self.cross_attention(cat_stack, cat_stack, cat_stack, need_weights=False)
            cat_final = cat_attended.flatten(1)
        else:
            cat_final = torch.zeros(batch_size, 0, device=x.device)

        # Continuous embedding
        if self.cont_embedding is not None and len(self.cont_indices) > 0:
            cont_vals = x[:, self.cont_indices]
            cont_emb = self.cont_embedding(cont_vals)
        else:
            cont_emb = torch.zeros(batch_size, 0, device=x.device)

        combined = torch.cat([cat_final, cont_emb], dim=-1) if cont_emb.numel() > 0 else cat_final
        return self.output_proj(combined)


class MultiScaleCategoricalAttention(nn.Module):
    """
    Multi-scale attention: local (within-category) + global (cross-category) heads.
    Key innovation: different attention heads specialize in different dependency scales.
    """
    def __init__(self, emb_dim: int, num_heads: int = 4):
        super().__init__()
        assert num_heads >= 2 and num_heads % 2 == 0, "num_heads must be even and >=2"

        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        local_heads = num_heads // 2
        global_heads = num_heads // 2

        self.local_attn = nn.MultiheadAttention(emb_dim, local_heads, dropout=0.1, batch_first=False)
        self.global_attn = nn.MultiheadAttention(emb_dim, global_heads, dropout=0.1, batch_first=False)

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, cat_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (seq_len, batch, emb_dim)
        cat_mask: (seq_len,) bool; True indicates categorical token
        """
        seq_len, batch_size, emb_dim = x.shape
        device = x.device

        # Local attention with optional categorical masking
        if cat_mask is not None and bool(cat_mask.any()):
            attn_mask = torch.zeros(seq_len, seq_len, device=device)
            non_cat = ~cat_mask
            attn_mask[non_cat][:, non_cat] = float("-inf")
            local_out, local_w = self.local_attn(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        else:
            local_out, local_w = self.local_attn(x, x, x, need_weights=True, average_attn_weights=False)

        global_out, global_w = self.global_attn(x, x, x, need_weights=True, average_attn_weights=False)

        # Fuse outputs
        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))

        # Combine attention weights
        local_heads = local_w.mean(dim=0)
        global_heads = global_w.mean(dim=0)
        A_heads = torch.cat([local_heads, global_heads], dim=0)

        return fused, A_heads


class EnhancedCPFA(nn.Module):
    """
    Controllable-Prior Feature Attention (CPFA) - Core SNI module.
    
    Key innovations:
    1. Multi-head attention over (d-1) feature tokens
    2. Learnable per-head confidence λ_h via softplus(θ_λ)
    3. Optional missingness-indicator embedding (SNI-M variant)
    4. Statistical prior integration with learnable strength
    """
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        num_heads: int,
        hidden_dims: List[int],
        output_dim: int,
        is_classification: bool = False,
        dropout: float = 0.1,
        cat_indices: Optional[List[int]] = None,
        use_cat_embedding: bool = True,
        use_multiscale: bool = True,
        mask_aware: bool = False,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.is_cls = bool(is_classification)
        self.cat_indices = list(cat_indices or [])
        self.use_multiscale = bool(use_multiscale and is_classification)
        self.mask_aware = bool(mask_aware)

        # Feature embedding
        self.feature_embedding = nn.Linear(1, self.emb_dim)

        # Missingness embedding (for SNI-M)
        if self.mask_aware:
            self.missing_embedding = nn.Linear(1, self.emb_dim)
        else:
            self.missing_embedding = None

        # Optional categorical embedding
        if use_cat_embedding and len(self.cat_indices) > 0:
            self.cat_embedding_module = CategoricalEmbeddingModule(input_dim, self.cat_indices, self.emb_dim, self.emb_dim // 2)
            self.fusion_layer = nn.Linear(self.emb_dim * 2, self.emb_dim)
        else:
            self.cat_embedding_module = None
            self.fusion_layer = None

        # Attention core
        if self.use_multiscale:
            self.attention = MultiScaleCategoricalAttention(self.emb_dim, self.num_heads)
            self.attn = None
        else:
            self.attention = None
            self.attn = nn.MultiheadAttention(
                embed_dim=self.emb_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=False,
            )

        # Output MLP
        mlp_layers: List[nn.Module] = []
        in_dim = self.emb_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, int(h)))
            mlp_layers.append(nn.LayerNorm(int(h)))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = int(h)
        mlp_layers.append(nn.Linear(in_dim, int(output_dim)))
        self.mlp = nn.Sequential(*mlp_layers)

        self.dropout = nn.Dropout(dropout)

        # Learnable per-head confidence (lambda)
        self.theta_lambda = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, Z: torch.Tensor, M: Optional[torch.Tensor] = None):
        """
        Z: (batch, d-1) source features
        M: (batch, d-1) missingness mask (1=observed, 0=missing) for SNI-M
        
        Returns:
            y_hat: predictions
            A_heads: (num_heads, seq_len, seq_len) attention weights
            A_avg_mat: averaged attention matrix
            lambdas: (num_heads,) confidence parameters
        """
        batch_size, d_minus1 = Z.size()
        device = Z.device

        # Embed features
        Z_reshaped = Z.contiguous().view(-1, 1)
        E = self.feature_embedding(Z_reshaped).view(batch_size, d_minus1, self.emb_dim)

        # Add missingness embedding if applicable
        if self.mask_aware:
            assert self.missing_embedding is not None
            if M is None:
                raise ValueError("mask_aware=True but M is None")
            M_reshaped = M.contiguous().view(-1, 1)
            Em = self.missing_embedding(M_reshaped).view(batch_size, d_minus1, self.emb_dim)
            E = E + Em

        # Fuse categorical embedding if available
        if self.cat_embedding_module is not None:
            cat_emb = self.cat_embedding_module(Z)
            cat_emb_expanded = cat_emb.unsqueeze(1).expand(-1, d_minus1, -1)
            E = self.fusion_layer(torch.cat([E, cat_emb_expanded], dim=-1))

        # Attention (seq_first for MultiheadAttention)
        E = E.permute(1, 0, 2)  # (seq, batch, emb)

        if self.use_multiscale and self.attention is not None:
            cat_mask = torch.zeros(d_minus1, dtype=torch.bool, device=device)
            if len(self.cat_indices) > 0:
                for idx in self.cat_indices:
                    if idx < d_minus1:
                        cat_mask[idx] = True
            attn_out, A_heads = self.attention(E, cat_mask)
        else:
            assert self.attn is not None
            attn_out, attn_w = self.attn(E, E, E, need_weights=True, average_attn_weights=False)
            A_heads = attn_w.mean(dim=0)  # (num_heads, seq, seq)

        # Compute lambdas
        lambdas = F.softplus(self.theta_lambda)

        # Average attention matrix
        A_avg_mat = A_heads.mean(dim=0)

        # Pool and predict
        out_feat = attn_out.mean(dim=0)  # (batch, emb)
        out_feat = self.dropout(out_feat)
        y_hat = self.mlp(out_feat)

        return y_hat, A_heads, A_avg_mat, lambdas

    @staticmethod
    def compute_prior_loss(A_vec_heads: torch.Tensor, P_f: torch.Tensor, lambdas: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute prior regularization loss.
        
        A_vec_heads: (H, d_minus1) head-level source-importance distributions
        P_f: (d_minus1,) statistical prior
        lambdas: (H,) per-head confidence
        alpha: global prior strength (decays over EM iterations)
        """
        H = lambdas.size(0)
        if A_vec_heads.dim() != 2 or A_vec_heads.size(0) != H:
            raise ValueError(f"Expected A_vec_heads shape (H,d), got {tuple(A_vec_heads.shape)} with H={H}")
        P_rep = P_f.unsqueeze(0).repeat(H, 1)
        diff = A_vec_heads - P_rep
        sq_norm = torch.sum(diff * diff, dim=1)
        return float(alpha) * torch.sum(lambdas * sq_norm)


class DualPathCPFA(EnhancedCPFA):
    """
    Dual-path CPFA for categorical targets.
    
    Innovation: Two processing paths combined via learned gating:
    - Path 1: Base attention-based classification
    - Path 2: Continuous embedding path for smoother gradients
    """
    def __init__(self, *args, **kwargs):
        hidden_dims = kwargs.get("hidden_dims", [64, 32])
        output_dim = kwargs.get("output_dim", 2)
        dropout = kwargs.get("dropout", 0.1)
        super().__init__(*args, **kwargs)

        if self.is_cls:
            self.continuous_path = nn.Sequential(
                nn.Linear(self.emb_dim, int(hidden_dims[0])),
                nn.LayerNorm(int(hidden_dims[0])),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_dims[0]), int(hidden_dims[-1])),
                nn.ReLU(),
                nn.Linear(int(hidden_dims[-1]), int(output_dim)),
            )
            self.path_gate = nn.Sequential(
                nn.Linear(self.emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1),
            )
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, Z: torch.Tensor, M: Optional[torch.Tensor] = None):
        y_hat_base, A_heads, A_avg_mat, lambdas = super().forward(Z, M=M)
        if not self.is_cls:
            return y_hat_base, A_heads, A_avg_mat, lambdas

        # Recompute features for gating
        batch_size, d_minus1 = Z.size()
        Z_reshaped = Z.contiguous().view(-1, 1)
        E = self.feature_embedding(Z_reshaped).view(batch_size, d_minus1, self.emb_dim)

        if self.mask_aware:
            assert self.missing_embedding is not None
            if M is None:
                raise ValueError("mask_aware=True but M is None")
            M_reshaped = M.contiguous().view(-1, 1)
            Em = self.missing_embedding(M_reshaped).view(batch_size, d_minus1, self.emb_dim)
            E = E + Em

        if self.cat_embedding_module is not None:
            cat_emb = self.cat_embedding_module(Z)
            cat_emb_expanded = cat_emb.unsqueeze(1).expand(-1, d_minus1, -1)
            E = self.fusion_layer(torch.cat([E, cat_emb_expanded], dim=-1))

        E = E.permute(1, 0, 2)
        if self.use_multiscale and self.attention is not None:
            cat_mask = torch.zeros(d_minus1, dtype=torch.bool, device=Z.device)
            if len(self.cat_indices) > 0:
                for idx in self.cat_indices:
                    if idx < d_minus1:
                        cat_mask[idx] = True
            attn_out, _ = self.attention(E, cat_mask)
        else:
            assert self.attn is not None
            attn_out, _ = self.attn(E, E, E, need_weights=False)
        out_feat = self.dropout(attn_out.mean(dim=0))

        cont_logits = self.continuous_path(out_feat)
        gate = self.path_gate(out_feat)
        logits = (gate[:, 0:1] * y_hat_base + gate[:, 1:2] * cont_logits) / self.temperature
        return logits, A_heads, A_avg_mat, lambdas


class CategoricalCPFATrainer:
    """
    Trainer utilities for categorical CPFA with advanced techniques.
    
    Features:
    - Focal loss for class imbalance
    - Label smoothing for regularization  
    - Mixup data augmentation
    - Curriculum learning
    """
    def __init__(
        self,
        use_focal_loss: bool = True,
        use_label_smoothing: bool = True,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        use_curriculum: bool = True,
    ):
        self.use_focal_loss = bool(use_focal_loss)
        self.use_label_smoothing = bool(use_label_smoothing)
        self.use_mixup = bool(use_mixup)
        self.mixup_alpha = float(mixup_alpha)
        self.use_curriculum = bool(use_curriculum)

    @staticmethod
    def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def create_loss_function(self, class_weights: np.ndarray, n_classes: int, epsilon: float = 0.1) -> nn.Module:
        if self.use_focal_loss:
            alpha = torch.tensor(class_weights, dtype=torch.float32)
            return FocalCrossEntropyLoss(alpha=alpha, gamma=2.0)
        if self.use_label_smoothing:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            return LabelSmoothingCrossEntropy(epsilon=epsilon, weight=weight)
        weight = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        P_f_tensor: torch.Tensor,
        alpha_g_minus_1: float,
        epoch: int,
        total_epochs: int,
        device: torch.device,
        mask_aware: bool = False,
        prior_mode: str = "learned",
        hard_prior_lambda: float = 10.0,
        lambda_mode: str = "learned",
        lambda_fixed_value: float = 1.0,
    ) -> float:
        model.train()
        total_loss = 0.0

        difficulty = min(1.0, epoch / (total_epochs * 0.5)) if self.use_curriculum else 1.0

        for batch in train_loader:
            if mask_aware:
                Xb, Mb, yb = batch
                Xb = Xb.to(device)
                Mb = Mb.to(device)
                yb = yb.to(device)
            else:
                Xb, yb = batch
                Xb = Xb.to(device)
                yb = yb.to(device)

            optimizer.zero_grad()

            if self.use_mixup and np.random.random() < difficulty:
                if mask_aware:
                    Xb_mix, y_a, y_b, lam = self.mixup_data(Xb, yb, self.mixup_alpha)
                    y_hat, A_heads, _, lambdas = model(Xb_mix, M=Mb)
                else:
                    Xb_mix, y_a, y_b, lam = self.mixup_data(Xb, yb, self.mixup_alpha)
                    y_hat, A_heads, _, lambdas = model(Xb_mix)

                A_vec_heads = A_heads.mean(dim=1)
                loss_recon = lam * criterion(y_hat, y_a) + (1 - lam) * criterion(y_hat, y_b)
            else:
                if mask_aware:
                    y_hat, A_heads, _, lambdas = model(Xb, M=Mb)
                else:
                    y_hat, A_heads, _, lambdas = model(Xb)

                A_vec_heads = A_heads.mean(dim=1)
                loss_recon = criterion(y_hat, yb)

            # v0.3: Support lambda_mode='fixed'
            if lambda_mode == "fixed":
                lambdas_eff = torch.full_like(lambdas, float(lambda_fixed_value))
            else:
                lambdas_eff = lambdas

            if alpha_g_minus_1 <= 0 or prior_mode == "none":
                loss_prior = torch.tensor(0.0, device=device)
            elif prior_mode == "hard":
                lambdas_eff = torch.full_like(lambdas, float(hard_prior_lambda))
                loss_prior = model.compute_prior_loss(A_vec_heads, P_f_tensor, lambdas_eff, alpha_g_minus_1)
            else:
                loss_prior = model.compute_prior_loss(A_vec_heads, P_f_tensor, lambdas_eff, alpha_g_minus_1)

            loss = loss_recon + loss_prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * Xb.size(0)

        return total_loss / max(1, len(train_loader.dataset))
