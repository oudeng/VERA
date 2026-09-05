"""P5R-H SS0/SS1: terminology bridge for frozen data keys.

Frozen run artifacts and analysis outputs carry historical class keys.
Generators render those keys ONLY through this module, so no banned
literal appears in any generator's source strings (facts_gate layer 2).
The mapping itself is declared in docs/terminology_registry.json.
"""
from __future__ import annotations

import json
from pathlib import Path

_REG = Path(__file__).resolve().parent.parent / "docs" / "terminology_registry.json"


def _registry() -> dict:
    return json.loads(_REG.read_text())


def data_display(key: str) -> str:
    """Display term for a frozen data key (identity for unmapped keys)."""
    m = _registry()["data_key_display"]
    return m.get(key, key)


def condition_order() -> list:
    """The leakage condition keys in table order (raw data keys)."""
    return list(_registry()["condition_order"])


def banned_variants() -> list:
    """All banned variant strings (for registry-driven selftests)."""
    reg = _registry()
    out = []
    for term in reg["terms"]:
        out += term.get("banned", [])
        out += term.get("banned_word", [])
    return out
