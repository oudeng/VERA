"""Determinism control for code_SNI.

Implements engineering principle E3 from the P1 instruction: determinism is ON by
default, and the R0 behavior of silently disabling it on GPU is turned into an
explicit, recorded opt-in.

Background (P0 finding B10)
---------------------------
In R0, ``SNI_v0_3/utils.py:15-31`` set ``cudnn.deterministic = True`` and
``cudnn.benchmark = False``, but ``SNI_v0_3/imputer.py:240-242`` then called
``enable_performance_mode()`` whenever ``cfg.use_gpu`` was true, which reversed
both flags and additionally enabled TF32. Every SNI manifest had ``use_gpu=True``,
so no R0 SNI result is bit-reproducible. ``enable_torch_determinism()`` existed in
the R0 tree but was never called from anywhere.

Here the two modes are separated, neither is applied implicitly, and whichever is
chosen is written into the run's ``run_config.json`` (see ``common.runconfig``).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

#: Mode names accepted by :func:`apply`.
MODES = ("deterministic", "performance")


@dataclass(frozen=True)
class DeterminismState:
    """Exactly what was applied, for recording in ``run_config.json``."""

    mode: str
    seed: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    tf32_matmul: bool
    tf32_cudnn: bool
    torch_deterministic_algorithms: bool
    torch_num_threads: Optional[int]
    cublas_workspace_config: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

#: The value ``PYTHONHASHSEED`` actually had when this interpreter started.
#:
#: Captured at import, because :func:`seed_everything` below overwrites the
#: environment variable with the per-run seed. That assignment is a faithful port
#: of R0 (`utils.py:15-31`) and must stay -- but it has **no effect on the running
#: interpreter** (finding B48), while it does change what a later reader sees. A
#: run started with ``PYTHONHASHSEED=2025`` and seeded with ``seed=1`` would
#: otherwise record "1" in its run_config, and anyone reproducing from that
#: record would set the wrong value and fail. Recording the startup value is the
#: only honest option.
STARTUP_PYTHONHASHSEED: Optional[str] = os.environ.get("PYTHONHASHSEED")



def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch (CPU and CUDA).

    Byte-identical to R0 ``SNI_v0_3/utils.py:15-31`` except that the cudnn flags
    are *not* set here — flag policy belongs to :func:`apply` so that it is a
    single, recorded decision rather than something two functions fight over.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def apply(
    mode: str = "deterministic",
    *,
    seed: int,
    num_threads: Optional[int] = None,
    strict: bool = False,
) -> DeterminismState:
    """Seed everything and apply one explicit determinism policy.

    Parameters
    ----------
    mode
        ``"deterministic"`` (default) — reproducible; cudnn deterministic, no
        benchmark autotuning, TF32 off, and torch's deterministic algorithm
        checks enabled.
        ``"performance"`` — the R0 GPU behavior, kept only so that R0 runs can be
        re-created for the equivalence check. Never selected implicitly.
    seed
        Global seed.
    num_threads
        If given, cap torch intra-op threads (batch runners use 1).
    strict
        Only meaningful for ``"deterministic"``. When True, torch raises on any
        op lacking a deterministic implementation instead of warning.

    Returns
    -------
    DeterminismState
        Record it in ``run_config.json``; do not reconstruct it by guessing.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    seed_everything(seed)

    if mode == "deterministic":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # Required for deterministic cuBLAS GEMM on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        det_algos = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception:
            det_algos = False
    else:  # performance — reproduces R0's effective GPU settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        det_algos = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass

    return DeterminismState(
        mode=mode,
        seed=int(seed),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        tf32_matmul=bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        tf32_cudnn=bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        torch_deterministic_algorithms=det_algos,
        torch_num_threads=int(num_threads) if num_threads is not None else None,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
