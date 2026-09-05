"""Variable-role audit table (P5R SS0.2, internal review SS4.8): for every
dataset, the downstream target's role across the four layers -- config
declaration, missingness config, actual mask files, imputer feature
schema -- plus the historical verification over all grid runs. Emits an
ESM fragment and a machine-readable JSON with evidence paths.

    PYTHONHASHSEED=2025 python reporting/audit_target_roles.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import (TableStyle, code_cell,  # noqa: E402
                             dataframe_to_tex, write_tex)

GRID = CODE_ROOT / "results" / "P2_main_grid"


def _mask_paths(cfg: dict, name: str, path=()):
    hits = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == name:
                hits.append(" > ".join(path + (k,)))
            hits += _mask_paths(v, name, path + (str(k),))
    return hits


def audit(code_root: Path = CODE_ROOT, grid: Path = GRID) -> dict:
    from baselines.schema import DataSchema
    dz = yaml.safe_load((code_root / "configs" / "datasets.yaml"
                         ).read_text())["datasets"]
    mz = yaml.safe_load((code_root / "configs" / "missingness.yaml"
                        ).read_text())
    rows = []
    for ds, stanza in dz.items():
        tgt = stanza.get("downstream_target")
        sch = DataSchema.from_yaml(code_root / "configs" / "datasets.yaml", ds)
        feats = list(sch.categorical_vars) + list(sch.continuous_vars)
        mp = [p for p in _mask_paths(mz, tgt) if f"> {ds} >" in p] if tgt else []
        masked = {}
        mdir = code_root / "data" / "masks" / "clinical_v1_shuffled" / ds
        cols = list(pd.read_csv(code_root / "data" / "derived_shuffled"
                                / f"{ds}_complete.csv", nrows=1).columns)
        ti = cols.index(tgt) if tgt in cols else None
        for m in sorted(mdir.glob(f"{ds}_*_mask.npy")):
            if ti is not None:
                masked[m.name] = int(np.load(m)[:, ti].sum())
        rows.append({"dataset": ds, "target": tgt,
                     "in_imputer_features": tgt in feats,
                     "n_features": len(feats),
                     "missingness_config_paths": mp,
                     "mask_files_masking_target":
                         {k: v for k, v in masked.items() if v > 0},
                     "always_observed": stanza.get("always_observed", [])})
    # historical verification over every grid run
    featsets = {}
    for ds in dz:
        sch = DataSchema.from_yaml(code_root / "configs" / "datasets.yaml", ds)
        featsets[ds] = set(list(sch.categorical_vars)
                           + list(sch.continuous_vars))
    viol = []
    n_runs = 0
    for ms in sorted(grid.glob("*/metrics_summary.json")):
        d = json.loads(ms.read_text())
        n_runs += 1
        if dz[d["dataset"]].get("downstream_target") in featsets[d["dataset"]]:
            viol.append(ms.parent.name)
    return {"per_dataset": rows, "n_runs_checked": n_runs,
            "violations": viol}


def build(out_tex: Path, out_json: Path, code_root: Path = CODE_ROOT,
          grid: Path = GRID) -> Path:
    a = audit(code_root, grid)
    if a["violations"]:
        raise ValueError(f"target-in-features violations: "
                         f"{a['violations'][:5]} -- refuse to emit the "
                         f"all-clear table")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(a, indent=1))
    body = pd.DataFrame([{
        "Dataset": r["dataset"],
        "Downstream target": code_cell(r["target"]),
        "Imputer feature": "no" if not r["in_imputer_features"] else
                           r"\textbf{YES}",
        "Masked in mask files": ("yes (dropped before use)"
                                 if r["mask_files_masking_target"] else "no"),
        "In missingness config": ("; ".join(
            p.split(" > ")[-3] for p in r["missingness_config_paths"])
            or "no")} for r in a["per_dataset"]])
    bit = ""
    bit_json = CODE_ROOT / "results" / "T5_stats" / "mask_bitcheck.json"
    if bit_json.exists():
        b = json.loads(bit_json.read_text())
        if b.get("verdict") == "CONSUMED-FEATURES BIT-IDENTICAL":
            bit = (rf" Generic table-shaped target mask columns in the "
                   rf"remaining datasets are written by the common mask "
                   rf"writer but are excluded before any schema-dependent "
                   rf"operation; because no target-specific configuration "
                   rf"branch and no shared RNG stream is involved there, "
                   rf"the intervention below addresses all pathways "
                   rf"capable of changing consumed feature masks, and a "
                   rf"digest manifest over every frozen mask file "
                   rf"(\texttt{{mask\_digest\_manifest.json}}) certifies "
                   rf"that none of them has moved. "
                   rf"A configuration-level bit-check regenerates all "
                   rf"{b['n_conditions']} affected masks (four datasets, "
                   rf"MAR/MNAR, three rates) with the target's missingness "
                   rf"rows deleted and the target removed from the maskable "
                   rf"set: every consumed feature column is bit-identical "
                   rf"to the frozen mask, the row permutation and the "
                   rf"per-column RNG stream keys are unchanged, and no "
                   rf"label is touched (\texttt{{mask\_bitcheck.json}}).")
    style = TableStyle(environment="table", notes=(
        rf"Machine-readable audit over the executed configuration and all "
        rf"{a['n_runs_checked']:,} grid runs: no dataset's downstream "
        rf"target is part of any imputer's feature schema (0 violations). "
        rf"Where the missingness configuration lists the target as a "
        rf"self-censoring column, the table-shaped mask file does carry a "
        rf"mask column for it; the pipeline restricts every mask to the "
        rf"feature schema before any imputer or metric consumes it, so "
        rf"that column is inert, and a runtime assertion now guards the "
        rf"restriction." + bit + rf" Evidence paths per dataset are in the "
        rf"machine-readable audit artifact named in the provenance "
        rf"header.",))
    tex = dataframe_to_tex(
        body, caption=(r"Variable-role audit: downstream targets across "
                       r"the configuration, mask, and feature layers."),
        label="tab:esm_target_roles", column_format="llccc",
        header=["Dataset", "Downstream target", "Imputer feature",
                "Masked in mask files", "In missingness config"],
        style=style, escape_data=False)
    return write_tex(out_tex, tex, provenance={
        "generator": "reporting/audit_target_roles.py",
        "input": f"configs/datasets.yaml + configs/missingness.yaml + "
                 f"data/masks + {grid}",
        "audit_json": str(out_json),
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    a = audit()
    check(a["n_runs_checked"] == 2565 and a["violations"] == [],
          "real audit: 2565 runs, zero target-in-features violations")
    four = {r["dataset"]: r for r in a["per_dataset"]}
    check(bool(four["CDC2022"]["mask_files_masking_target"])
          and not four["CDC2022"]["in_imputer_features"],
          "CDC2022: target masked in files yet outside the feature schema "
          "(case three)")
    check(any("MAR" in p and "MNAR" not in p
              for p in four["ComCri"]["missingness_config_paths"]),
          "ComCri: target present under MAR as well (as the review found)")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(CODE_ROOT / "reporting" / "out" / "sec_esm_target_roles.tex",
                CODE_ROOT / "results" / "target_role_audit.json")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
