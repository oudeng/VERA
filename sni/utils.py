# utils.py - Utility functions for SNI
# SNI v0.3

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python/NumPy/PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Determinism flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def enable_torch_determinism(strict: bool = False) -> None:
    """
    Force deterministic algorithms in PyTorch.
    """
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception:
            pass


def set_num_threads(n: int = 1) -> None:
    """
    Limit torch intra-op parallelism.
    """
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def enable_performance_mode() -> None:
    """
    Enable performance optimizations for inference/training.
    Call this when reproducibility is not critical.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 for Ampere GPUs (faster with minimal precision loss)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@dataclass
class DeviceConfig:
    use_gpu: bool = False

    def torch_device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
