"""Per-run configuration snapshots (engineering principle E2).

P0 finding B22: in R0 the baseline runner wrote ``run_config.json`` per run, but
the SNI runner (``scripts/run_manifest_parallel.py``) wrote none, so the
hyperparameters of an SNI run could not be recovered from its output directory —
you had to go back to the manifest CSV and hope it had not changed. The
single-run script ``scripts/run_experiment.py:336-348`` already dumped
``cfg.__dict__``; that capability simply was never carried into the batch path.

Here every run — SNI or baseline — writes one ``run_config.json`` containing the
full parameter set, the resolved seeds, the determinism policy actually applied,
the device, library versions, and the git commit of code_SNI. A run directory is
self-describing.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_CODE_ROOT = Path(__file__).resolve().parent.parent


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so a snapshot never fails to serialise."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy array
        try:
            return obj.tolist()
        except Exception:
            pass
    return repr(obj)


def git_commit(repo: Optional[Path] = None) -> Optional[str]:
    """Return the current commit of code_SNI, with ``-dirty`` if uncommitted."""
    repo = repo or _CODE_ROOT
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def library_versions() -> Dict[str, Optional[str]]:
    """Versions as actually imported (not as advertised by stale dist-info)."""
    import importlib

    names = ["numpy", "pandas", "scipy", "sklearn", "torch", "statsmodels",
             "xgboost", "shap", "hyperimpute", "matplotlib"]
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = getattr(importlib.import_module(n), "__version__", "unknown")
        except Exception:
            out[n] = None
    return out


def device_info() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False}
    info: Dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = None
    return info


def build(
    *,
    exp_id: str,
    method: str,
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
    seeds: Mapping[str, Any],
    determinism: Any,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a complete run configuration snapshot.

    ``determinism`` should be the :class:`common.determinism.DeterminismState`
    returned by ``determinism.apply`` — the policy that was actually applied, not
    the one that was requested.
    """
    snap: Dict[str, Any] = {
        "schema_version": 1,
        "exp_id": exp_id,
        "method": method,
        "params": _jsonable(params),
        "inputs": _jsonable(inputs),
        "seeds": _jsonable(seeds),
        "determinism": _jsonable(determinism),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "libraries": library_versions(),
            "device": device_info(),
            "code_SNI_git_commit": git_commit(),
        },
    }
    if extra:
        snap["extra"] = _jsonable(extra)
    return snap


def write(outdir: Path, snapshot: Mapping[str, Any], filename: str = "run_config.json") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
    return path
