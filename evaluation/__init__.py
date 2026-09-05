"""Evaluation layer for code_SNI.

* :mod:`evaluation.metrics`  -- R0-identical metrics plus an exclude-columns mode
* :mod:`evaluation.protocol` -- within-fold ``fit -> transform`` protocol and the
  fold-independent fallback for transductive imputers (reviewer point R1-3)

The metric definitions themselves live in :mod:`sni.metrics`, a byte-identical
port of the R0 implementation.  Nothing in this package redefines them.
"""

from . import metrics, protocol  # noqa: F401

__all__ = ["metrics", "protocol"]
