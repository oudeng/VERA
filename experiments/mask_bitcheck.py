"""P5R-F SS4.2 / second internal review SS5.2: bit-level mask verification.

Claim under proof: the downstream target's presence in the missingness
configuration has NO influence on any consumed feature's mask. Datasets
where the target appears in the executed config as a masked column:
CDC2022 (MNAR), AutoMPG (MNAR), ComCri (MAR+MNAR), Concrete (MNAR).

Intervention (config level, zero training): the target's explicit config
row is deleted AND the target is added to `always_observed`, i.e. the
target is removed from the maskable set entirely. (Literal row deletion
alone would fall back to the profile's `default` entry where one exists,
so the executed intervention is the maximal reading of "delete the
target's config rows".) The masks are then regenerated with the frozen
metadata's recorded seed and row-order settings and compared to the
frozen mask files.

Phase A per condition -- current config: every feature-schema column
must regenerate bit-identical to the frozen mask. The target column
regenerates all-observed because the B2 forward fix (commit df46da6)
makes resolve() auto-exclude the declared downstream target; the frozen
files predate that fix and carry a masked target column, which the
pipeline drops before any consumer (runtime-asserted in run_grid).
Phase B per condition -- deletion on top: same feature bit-identity with
the target's config rows removed; rowspace digest unchanged; per-column
stream entropies for shared (purpose, mechanism, column) keys unchanged.
Labels are never touched: the intervention edits configuration only, and
the comparison is on mask arrays over the same immutable complete table.

    env PYTHONHASHSEED=2025 python experiments/mask_bitcheck.py
    env PYTHONHASHSEED=2025 python experiments/mask_bitcheck.py --selftest
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from missingness.generator import _rate_tag, generate_from_config  # noqa: E402
from baselines.schema import DataSchema  # noqa: E402

FROZEN = CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
OUT = CODE_ROOT / "results" / "T5_stats" / "mask_bitcheck.json"
RATES = [0.1, 0.3, 0.5]
#: dataset -> mechanisms whose executed config lists the target as a masked column
AFFECTED = {"CDC2022": ["MNAR"], "AutoMPG": ["MNAR"],
            "ComCri": ["MAR", "MNAR"], "Concrete": ["MNAR"]}


def _table(ds: str, dcfg: dict) -> pd.DataFrame:
    blk = dcfg["datasets"][ds]
    for key in ("complete_path", "source_path_R0"):
        p = blk.get(key)
        if p and (CODE_ROOT / p).exists():
            return pd.read_csv(CODE_ROOT / p)
    raise FileNotFoundError(ds)


def _deleted_config(cfg: dict, ds: str, target: str) -> dict:
    """Remove the target from the maskable set of dataset `ds` (all mechanisms)."""
    mod = copy.deepcopy(cfg)
    n_rows_deleted = 0
    for prof in mod.get("profiles", {}).values():
        blk = (prof.get("datasets") or {}).get(ds)
        if not blk:
            continue
        common = blk.setdefault("common", {})
        ao = list(common.get("always_observed", []) or [])
        if target not in ao:
            common["always_observed"] = ao + [target]
        for mech_blk in blk.values():
            if not isinstance(mech_blk, dict):
                continue
            for sect in ("mar", "mnar"):
                cols = (mech_blk.get(sect) or {}).get("columns") or {}
                if target in cols:
                    del cols[target]
                    n_rows_deleted += 1
    mod["_bitcheck_rows_deleted"] = n_rows_deleted
    return mod


def _streams_by_key(meta: dict) -> dict:
    return {tuple([s["purpose"]] + [str(x) for x in s["key"]]): s["entropy"]
            for s in (meta.get("rng", {}).get("streams") or [])}


def run_one(ds: str, mech: str, rate: float, cfg_pristine_path: Path,
            cfg_deleted_path: Path, dcfg: dict) -> dict:
    blk = dcfg["datasets"][ds]
    target = blk["downstream_target"]
    stem = f"{ds}_{mech}_{_rate_tag(rate)}"
    frozen_mask = np.load(FROZEN / ds / f"{stem}_mask.npy").astype(bool)
    fmeta = json.loads((FROZEN / ds / f"{stem}_meta.json").read_text())
    seed = int(fmeta["rng"]["root_seed"])
    ro = fmeta.get("row_order", {})
    overrides = {"row_order": {"mode": ro.get("mode", "as_is"),
                               "seed": ro.get("seed")}}

    df = _table(ds, dcfg)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)

    def _gen(cfg_path: Path):
        res = generate_from_config(
            df, ds, mech, rate, profile=fmeta["spec"]["profile"], seed=seed,
            config_path=cfg_path, overrides=overrides)
        cols = list(res.X_missing.columns)
        return res.mask_bool, cols, res.meta

    # Phase A: current pristine config. Since the B2 forward fix (commit
    # df46da6) resolve() auto-excludes the declared downstream target from
    # masking, so the target column regenerates all-observed while the frozen
    # file (generated pre-B2) carries a masked target column that the
    # pipeline drops before use. Feature columns must reproduce bit-for-bit.
    m0, cols0, meta0 = _gen(cfg_pristine_path)
    fidx = [cols0.index(c) for c in feats if c in cols0]
    tidx = cols0.index(target)
    reproA_feats = bool(np.array_equal(m0[:, fidx], frozen_mask[:, fidx]))
    targetA_observed = bool((~m0[:, tidx]).all())
    frozen_target_masked = int(frozen_mask[:, tidx].sum())

    # Phase B: target's explicit config rows deleted as well.
    m1, cols1, meta1 = _gen(cfg_deleted_path)
    assert cols1 == cols0, "column order changed by the config edit"
    feats_identical = bool(np.array_equal(m1[:, fidx], frozen_mask[:, fidx]))
    n_feature_bits_differ = int((m1[:, fidx] != frozen_mask[:, fidx]).sum())
    target_all_observed = bool((~m1[:, tidx]).all())
    row_digest_same = (meta1.get("row_order", {}).get("rowspace_digest")
                       == fmeta.get("row_order", {}).get("rowspace_digest"))
    s_f, s_1 = _streams_by_key(fmeta), _streams_by_key(meta1)
    shared = sorted(set(s_f) & set(s_1))
    streams_same = all(s_f[k] == s_1[k] for k in shared)
    return {"dataset": ds, "mechanism": mech, "rate": rate, "target": target,
            "phaseA_features_bit_identical": reproA_feats,
            "phaseA_target_auto_excluded": targetA_observed,
            "frozen_target_column_n_masked": frozen_target_masked,
            "features_bit_identical_after_deletion": feats_identical,
            "n_feature_bits_differ": n_feature_bits_differ,
            "n_feature_columns": len(fidx),
            "target_column_all_observed_after_deletion": target_all_observed,
            "rowspace_digest_unchanged": row_digest_same,
            "n_shared_streams": len(shared),
            "shared_stream_entropies_identical": bool(streams_same)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    cfg = yaml.safe_load((CODE_ROOT / "configs" / "missingness.yaml").read_text())
    dcfg = yaml.safe_load((CODE_ROOT / "configs" / "datasets.yaml").read_text())
    with tempfile.TemporaryDirectory() as td:
        pristine = Path(td) / "missingness_pristine.yaml"
        pristine.write_text((CODE_ROOT / "configs" / "missingness.yaml").read_text())
        rows, deleted_counts = [], {}
        for ds, mechs in AFFECTED.items():
            target = dcfg["datasets"][ds]["downstream_target"]
            mod = _deleted_config(cfg, ds, target)
            deleted_counts[ds] = mod.pop("_bitcheck_rows_deleted")
            dpath = Path(td) / f"missingness_del_{ds}.yaml"
            dpath.write_text(yaml.safe_dump(mod, sort_keys=False))
            for mech in mechs:
                for rate in RATES:
                    r = run_one(ds, mech, rate, pristine, dpath, dcfg)
                    rows.append(r)
                    ok = (r['features_bit_identical_after_deletion']
                          and r['phaseA_features_bit_identical'])
                    print(f"[{'OK' if ok else 'FAIL'}] "
                          f"{ds} {mech}@{rate:g} featsA={r['phaseA_features_bit_identical']} "
                          f"featsB={r['features_bit_identical_after_deletion']} "
                          f"({r['n_feature_columns']} cols) target_unmasked="
                          f"{r['target_column_all_observed_after_deletion']}")

    all_ok = (all(r["phaseA_features_bit_identical"] for r in rows)
              and all(r["features_bit_identical_after_deletion"] for r in rows)
              and all(r["phaseA_target_auto_excluded"] for r in rows)
              and all(r["target_column_all_observed_after_deletion"] for r in rows)
              and all(r["rowspace_digest_unchanged"] for r in rows)
              and all(r["shared_stream_entropies_identical"] for r in rows))
    out = {"claim": "the downstream target's presence in the missingness "
                    "configuration has no influence on any consumed feature's "
                    "mask: every feature column regenerates bit-identically "
                    "with the target's config rows present or deleted "
                    "(name-keyed per-column RNG streams)",
           "intervention": "phase A: current config unchanged (resolve() "
                           "auto-excludes the declared target since the B2 "
                           "forward fix, commit df46da6; frozen masks predate "
                           "it and carry an inert target mask column that the "
                           "pipeline drops before use, runtime-asserted); "
                           "phase B: additionally, the target's explicit "
                           "mar/mnar rows deleted and the target added to "
                           "always_observed",
           "config_rows_deleted": deleted_counts,
           "n_conditions": len(rows), "conditions": rows,
           "verdict": "CONSUMED-FEATURES BIT-IDENTICAL" if all_ok else "MISMATCH"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("config_rows_deleted", "n_conditions", "verdict")}))
    return 0 if all_ok else 1


def _selftest() -> int:
    """_deleted_config: removes explicit rows, adds always_observed, touches nothing else."""
    cfg = {"profiles": {"p": {"datasets": {"D": {
        "common": {"always_observed": ["ID"]},
        "MNAR": {"mnar": {"default": {"mode": "logit"},
                          "columns": {"tgt": {"mode": "logit"}, "x": {"mode": "logit"}}}},
        "MAR": {"mar": {"columns": {"tgt": {"drivers": ["x"]}}}}}}}}}
    mod = _deleted_config(cfg, "D", "tgt")
    ok = True
    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and c
    check(mod["_bitcheck_rows_deleted"] == 2, "both explicit target rows deleted")
    d = mod["profiles"]["p"]["datasets"]["D"]
    check("tgt" in d["common"]["always_observed"], "target now always_observed")
    check("tgt" not in d["MNAR"]["mnar"]["columns"], "mnar row gone")
    check("x" in d["MNAR"]["mnar"]["columns"], "other columns untouched")
    check("tgt" not in d["MAR"]["mar"]["columns"], "mar row gone")
    check(cfg["profiles"]["p"]["datasets"]["D"]["MNAR"]["mnar"]["columns"].get("tgt")
          is not None, "original config object not mutated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
