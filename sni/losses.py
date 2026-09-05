# losses.py - Custom loss functions for SNI
# SNI v0.3

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageLoss(nn.Module):
    """
    Piecewise Huber + Quantile loss for continuous prediction.
    Provides robustness to outliers with adaptive loss scaling.
    """
    def __init__(self, delta_mid: float = 1.0, delta_tail: float = 5.0, tail_threshold: float = 2.0, quantile: float = 0.5):
        super().__init__()
        self.delta_mid = float(delta_mid)
        self.delta_tail = float(delta_tail)
        self.tail_threshold = float(tail_threshold)
        self.quantile = float(quantile)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        diff = (pred - true).abs().flatten()
        mid_mask = (diff <= self.tail_threshold)

        loss_mid = torch.where(
            mid_mask,
            0.5 * (diff ** 2) / self.delta_mid,
            self.delta_mid * (diff - 0.5 * self.delta_mid),
        )
        loss_tail = torch.where(
            ~mid_mask,
            0.5 * (diff ** 2) / self.delta_tail,
            self.delta_tail * (diff - 0.5 * self.delta_tail),
        )
        huber_loss = torch.where(mid_mask, loss_mid, loss_tail).mean()

        residual = (true - pred).flatten()
        q = self.quantile
        q_loss = torch.max(q * residual, (q - 1) * residual).mean()
        return huber_loss + q_loss


class FocalCrossEntropyLoss(nn.Module):
    """
    Focal loss for handling class imbalance in categorical prediction.
    Down-weights easy examples to focus on hard cases.
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
            focal = at * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing for regularization.
    Prevents overconfident predictions.
    """
    def __init__(self, epsilon: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (n - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.epsilon)

        if self.weight is not None:
            w = self.weight.to(pred.device)
            weights = w.gather(0, target)
            loss = (-true_dist * log_probs).sum(dim=1) * weights
        else:
            loss = (-true_dist * log_probs).sum(dim=1)

        return loss.mean()
