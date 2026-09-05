"""Every adjudication in docs/adjudications.md, as an executable assertion.

A ruling is not in force because it was written down. It is in force when a test
fails if the config drifts away from it.

The incident that produced this file: P2e §3.1 ruled SNI's canonical device to be
CPU. The ruling was accepted, a CPU schedule was published from it, and
`configs/scheduling.yaml` still said `SNI: gpu` hours later. `run_grid.py` and
`recompute_schedule.py` both read that key, so the full grid would have run on
CUDA while the report said CPU -- and nothing would have raised. That is B1's
class of defect (declared value != implemented value), committed by us, on our
own ruling.

Run this first when resuming work:

    env PYTHONHASHSEED=2025 python -m pytest tests/test_adjudications.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent


def cfg(name: str) -> dict:
    return yaml.safe_load((CODE_ROOT / "configs" / f"{name}.yaml").read_text())


# --------------------------------------------------------------------------- #
# A-1 — training protocol: early stopping off, 200 epochs as already configured
# --------------------------------------------------------------------------- #
def test_a1_training_protocol():
    p = cfg("training_protocol")["protocol"]
    assert p["disable_early_stopping"] is True
    # 200 is not a new setting -- it is the value already present in the code.
    # Changing it would turn "the budget was executed" into "the budget was
    # raised", which is exactly the accusation the ruling exists to avoid.
    assert p["epochs"]["SNI"] == 200
    assert p["epochs"]["TabCSDI"] == 200


def test_a1_affected_methods_audit_is_present():
    """GAIN and MIWAE must stay recorded as having no early stopping.

    B73 ("GAIN is genuinely weak") depends on this: if GAIN were also truncated by
    a stopping rule, Table S3's "SNI significantly beats GAIN" would need
    re-examining. The audit is the evidence, so it must not quietly disappear.
    """
    a = cfg("training_protocol")["affected_methods"]
    assert a["SNI"]["has_early_stopping"] is True
    assert a["TabCSDI"]["has_early_stopping"] is True
    assert a["GAIN"]["has_early_stopping"] is False
    assert a["MIWAE"]["has_early_stopping"] is False


# --------------------------------------------------------------------------- #
# A-2 — SNI's canonical device is CPU
# --------------------------------------------------------------------------- #
def test_a2_sni_on_cpu():
    placement = cfg("scheduling")["method_placement"]
    assert placement["SNI"] == "cpu", (
        "P2e §3.1 ruled SNI's canonical device to be CPU. This key governs "
        "run_grid.py and recompute_schedule.py; if it says gpu, the grid runs on "
        "CUDA while every report says CPU, and SNI's metrics are device-dependent "
        "(B83 -- the sign of MIMIC's R2 flips).")


def test_a2_tabcsdi_stays_on_gpu():
    """TabCSDI genuinely uses the device: 36.9 s on AutoMPG against SNI's 107.9 s."""
    assert cfg("scheduling")["method_placement"]["TabCSDI"] == "gpu"


def test_a2_methods_yaml_carries_no_device_policy():
    """P3-A ruling 1: one binding source for device policy, and it is scheduling.yaml.

    methods.yaml used to carry a `device:` block recording R0's de-facto
    assignment (`SNI: gpu_if_available`). The W4 ESM generator reads
    methods.yaml; had it sourced devices there, the ESM would print an R0
    record as an R1 declaration while the main text says CPU -- a fresh B1,
    manufactured by us. The record now lives in the corrections ledger (B2).
    """
    m = cfg("methods")
    offenders = [k for k in m if "device" in str(k).lower()]
    assert not offenders, (
        f"methods.yaml carries device-policy key(s) {offenders}; the binding "
        f"source is scheduling.yaml method_placement (ruling A-2), and history "
        f"belongs in the corrections ledger, not in a live config.")


def test_a2_esm_runtime_devices_come_from_scheduling():
    """The generated S6 device column must equal scheduling.yaml, method for method.

    Renaming the stray block only stops a human; this stops code (P3-A: '光改名
    只防住人，加断言才防住代码'). runtime_rows() is the exact function
    gen_runtime_frame() renders.
    """
    import sys
    sys.path.insert(0, str(CODE_ROOT))
    from reporting.esm_sections import runtime_rows

    placement = cfg("scheduling")["method_placement"]
    got = {r["Method"]: r["Device"].lower() for r in runtime_rows()}
    assert got == {m: d.lower() for m, d in placement.items()}, (
        "the ESM runtime table's device column no longer matches "
        "scheduling.yaml method_placement")
    assert got["SNI"] == "cpu"


# --------------------------------------------------------------------------- #
# A-3 — BLAS threads pinned before torch is imported, and verified
# --------------------------------------------------------------------------- #
def test_a3_threads_pinned_before_torch():
    """The pinning must textually precede the numpy/torch import, not follow it.

    Setting these after the import is a no-op, in the same way PYTHONHASHSEED is
    a no-op when set from inside the interpreter (B48).
    """
    src = (CODE_ROOT / "experiments" / "run_grid.py").read_text()
    pin = src.find("OMP_NUM_THREADS")
    imp = min(i for i in (src.find("\nimport numpy"), src.find("\nimport torch"),
                          src.find("\nimport pandas")) if i > 0)
    assert 0 < pin < imp, "thread pinning must precede the numpy/torch imports"


def test_a3_thread_count_is_verified_not_assumed():
    """An env var is a request; torch.get_num_threads() is the fact (B48's lesson)."""
    src = (CODE_ROOT / "experiments" / "run_grid.py").read_text()
    assert "get_num_threads" in src
    assert "REFUSING TO RUN" in src and "threads" in src.lower()


def test_a3_default_thread_count():
    src = (CODE_ROOT / "experiments" / "run_grid.py").read_text()
    m = re.search(r'SNI_NUM_THREADS["\']\s*,\s*["\'](\d+)["\']', src)
    assert m and m.group(1) == "2", (
        "12 CPU workers x 2 threads = 24 of 32 cores, leaving headroom for the "
        "GPU worker's host thread and I/O")


# --------------------------------------------------------------------------- #
# A-4 / A-5 — queue concurrency
# --------------------------------------------------------------------------- #
def test_a4_gpu_serial():
    assert cfg("scheduling")["policy"]["gpu_queue"]["max_concurrent"] == 1, (
        "B81: two concurrent GPU jobs measured 172x. Low GPU utilisation is why "
        "concurrency is catastrophic, not why it would be safe.")


def test_a5_cpu_workers():
    assert cfg("scheduling")["policy"]["cpu_queue"]["max_concurrent"] == 12, (
        "T2c.1: the ceiling comes from HyperImpute (0 of 16 completed), not KNN "
        "(still improving at 16).")


# --------------------------------------------------------------------------- #
# A-6 — the shuffled masks are the factory version
# --------------------------------------------------------------------------- #
def test_a6_shuffled_masks_are_factory():
    root = CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
    assert root.is_dir(), "the shipped masks are the shuffled ones (P2b decision 3)"
    datasets = list(cfg("datasets")["datasets"])
    missing = [d for d in datasets if not (root / d).is_dir()]
    assert not missing, f"no shuffled masks for {missing}"


def test_a6_rowspace_assertion_is_wired_in():
    """The one pairing error that produces no error must stay guarded."""
    src = (CODE_ROOT / "experiments" / "run_grid.py").read_text()
    assert "assert_same_rowspace" in src
    assert "md5" in src, "mask content hash guards B75 (regeneration mid-experiment)"


# --------------------------------------------------------------------------- #
# A-7 / A-8 — data layer
# --------------------------------------------------------------------------- #
def test_a7_no_target_as_feature():
    """A target column must never appear among the imputable features."""
    ds = cfg("datasets")["datasets"]
    for name, blk in ds.items():
        target = blk.get("target_column") or blk.get("target")
        if not target:
            continue
        imputable = [c for c, r in blk["columns"].items()
                     if r.get("role") == "imputable"]
        assert target not in imputable, f"{name}: target {target} is a feature"


def test_a8_mimic_table_shape():
    import pandas as pd
    p = CODE_ROOT / "data" / "derived_shuffled" / "MIMIC_complete.csv"
    if not p.exists():
        pytest.skip("MIMIC table not built in this checkout")
    d = pd.read_csv(p)
    assert len(d) == 2849, (
        "A-8 replaced R0's 2052x8 MIMIC with 2849x16; the 8/06 fallback gate "
        "passed all six criteria and ruled NOT to revert")


# --------------------------------------------------------------------------- #
# A-9 — CDC2022 row count
# --------------------------------------------------------------------------- #
def test_a9_cdc2022_rows():
    src = (CODE_ROOT / "experiments" / "budget_panel.py").read_text()
    m = re.search(r'ROWS\s*=\s*\{\s*["\']CDC2022["\']\s*:\s*(\d+)', src)
    assert m and m.group(1) == "1000", (
        "P2e §5: n=1000 is measured, where n=3000's cost interval spanned 7.7x. "
        "R2-4 asks about table width (d=41), which is unchanged.")


# --------------------------------------------------------------------------- #
# Cross-cutting: the failure policy that R0 violated
# --------------------------------------------------------------------------- #
def test_silent_skip_is_forbidden():
    """R0 lost 300 baselines_deep runs to a comment in an aggregation script."""
    fp = cfg("scheduling")["failure_policy"]
    assert fp["silent_skip"] == "forbidden"
    assert fp["on_run_failure"] == "write_error_log_and_continue"


def test_pythonhashseed_is_asserted_at_entry():
    """B48: it has no effect when set from inside, so entry must refuse without it."""
    for f in ("run_grid.py", "pilot_r21.py", "d_stability.py", "budget_panel.py"):
        src = (CODE_ROOT / "experiments" / f).read_text()
        assert "PYTHONHASHSEED" in src and "REFUSING TO RUN" in src, f
