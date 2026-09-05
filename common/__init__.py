"""Shared runtime infrastructure for code_SNI.

Implements the cross-cutting engineering principles from the P1 instruction:

* :mod:`common.determinism` -- E3, determinism on by default, mode recorded
* :mod:`common.runconfig`   -- E2, complete per-run configuration snapshots
* :mod:`common.masks`       -- E4, cached masks are loaded and verified
* :mod:`common.config`      -- E1, configs are the single source of truth
"""

from . import determinism, masks, runconfig  # noqa: F401
