"""The three tables, named so that a call site cannot confuse them.

A lineage audit on 2026-08-29 found two audit-family variants computed on the
pre-mask ground-truth table because a rule document called that file "the
same input the archived TAP used". The file is named `{ds}_complete.csv`,
where "complete" means complete-CASE; the sentence read as if it meant
"completed". The name is what caused the defect
(`data/derived_shuffled/README.md`, `docs/T53_input_correction_rules.md`).

This module makes the distinction impossible to lose at the call site. New
code should use these accessors rather than the literal paths; existing
reads are enumerated and classified by
`experiments/ground_truth_consumer_census.py`, which refuses an unregistered
one.

    from common.tables import ground_truth_table, masked_table, initial_completion
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
TABLES = CODE_ROOT / "data" / "derived_shuffled"
MASKS = CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"


def ground_truth_table(ds: str) -> pd.DataFrame:
    """The PRE-MASK benchmark table. Every value is a real observed value.

    Read this ONLY to (a) build a masked table, (b) obtain the withheld true
    values for scoring, (c) construct an injected leakage proxy, or (d) read
    column names and dtypes. Handing it to a model under test, an audit
    object or a comparator is the defect this module exists to prevent.
    """
    return pd.read_csv(TABLES / f"{ds}_complete.csv")


def feature_names(ds: str) -> Tuple[list, list, list]:
    """(categorical, continuous, all) feature names from the schema.

    The downstream target is not among them; that is what makes it
    unreachable by any imputer.
    """
    from baselines.schema import DataSchema
    sc = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    cat, cont = list(sc.categorical_vars), list(sc.continuous_vars)
    return cat, cont, cat + cont


def mask_frame(ds: str, mech: str = "MAR", rate: int = 30) -> pd.DataFrame:
    """The boolean mask over the feature columns: True where a cell is withheld."""
    G = ground_truth_table(ds)
    m = np.load(MASKS / ds / f"{ds}_{mech}_{rate}per_mask.npy").astype(bool)
    _, _, feats = feature_names(ds)
    return pd.DataFrame(m, columns=list(G.columns))[feats]


def masked_table(ds: str, mech: str = "MAR", rate: int = 30,
                 from_disk: bool = False) -> pd.DataFrame:
    """The table every imputer under test receives: withheld cells absent.

    `from_disk=False` (default) rebuilds it as the experiment drivers do,
    `complete[feats].mask(mask_df)`, restricted to the feature columns.
    `from_disk=True` reads the shipped CSV, which additionally carries the
    identifier and the downstream target. The two agree cell-for-cell on the
    feature columns; they differ only in the dtype label of never-masked
    integer columns after a CSV round-trip (verified in
    `results/T6_lineage/data_lineage_audit.json`).
    """
    if from_disk:
        return pd.read_csv(MASKS / ds / f"{ds}_{mech}_{rate}per.csv")
    _, _, feats = feature_names(ds)
    return ground_truth_table(ds)[feats].mask(mask_frame(ds, mech, rate))


def initial_completion(ds: str, seed: int = 1, mech: str = "MAR",
                       rate: int = 30) -> pd.DataFrame:
    """The table the first training round sees. Never written to disk.

    The masked table with the withheld cells filled by the imputer's own
    initial statistical completion -- sklearn's IterativeImputer (MICE),
    `max_iter=5`, `sample_posterior=False`, `initial_strategy="mean"`,
    `random_state=seed`, categoricals integer-coded and decoded around it
    (`sni/imputer.py::SNIImputer._initial_stat_impute`).

    This is the fair input: it contains no withheld value. It is what the
    archived TAP_0 was computed on, proven at the artifact level in
    `results/T6_lineage/data_lineage_audit.json`, and what the corrected
    T5.3 family variants are computed on.
    """
    from sni.imputer import SNIConfig, SNIImputer
    cat, cont, _ = feature_names(ds)
    imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.max_iters = 0        # EM never entered: impute() returns the
    #                              cast initial completion and trains nothing
    return imp.impute(X_missing=masked_table(ds, mech, rate),
                      X_complete=None, mask_df=mask_frame(ds, mech, rate))


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    for ds in ("MIMIC", "eICU"):
        G = ground_truth_table(ds)
        mk = mask_frame(ds)
        X = masked_table(ds)
        F = initial_completion(ds)
        _, _, feats = feature_names(ds)
        check(not G[feats].isna().to_numpy().any(),
              f"{ds}: the ground-truth table has no missing feature cell")
        check(int(X.isna().to_numpy().sum()) == int(mk.to_numpy().sum()),
              f"{ds}: the masked table is missing exactly the masked cells")
        check(not F.isna().to_numpy().any(),
              f"{ds}: the initial completion is dense")
        n = int(mk.to_numpy().sum())
        same = sum(int((pd.to_numeric(F[c], errors="coerce").to_numpy()[i]
                        == pd.to_numeric(G[c], errors="coerce").to_numpy()[i]))
                   for c in mk.columns
                   for i in np.where(mk[c].to_numpy())[0])
        check(same < n,
              f"{ds}: the initial completion is not the ground truth "
              f"({same}/{n} masked cells coincide)")
        disk = masked_table(ds, from_disk=True)
        common = [c for c in feats if c in disk.columns]
        def _txt(v):
            try:
                f = float(v)
                return "nan" if f != f else f.hex()
            except (TypeError, ValueError):
                return str(v)
        check(all(_txt(a) == _txt(b) for c in common
                  for a, b in zip(X[c].to_numpy(), disk[c].to_numpy())),
              f"{ds}: the rebuilt masked table equals the shipped one on "
              f"every feature cell")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    raise SystemExit(_selftest())
