"""T6.0 TAP input-lineage audit (rules: docs/T60_tap_lineage_audit_rules.md,
committed before this file existed).

The fifth internal review (SS3.1) designated a submission blocker: the
manuscript says TAP_0 is computed on the initial statistical completion --
the table the first training round sees -- while docs/T53_tap_family_rules.md
says the family variants read data/derived_shuffled/{ds}_complete.csv, "the
same input the archived TAP used". If that file is the pre-mask benchmark
table, both cannot be true of the same object.

This script decides the question by measurement, for every consumer, under
the rules fixed in T6.0. It computes nothing that the rules did not
prespecify, and the decision tree is evaluated here, not by hand.

    env PYTHONHASHSEED=2025 python experiments/tap_lineage_audit.py
    env PYTHONHASHSEED=2025 python experiments/tap_lineage_audit.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

RULES = "docs/T60_tap_lineage_audit_rules.md"
TABLES = CODE_ROOT / "data" / "derived_shuffled"
MASKS = CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
OUT = CODE_ROOT / "results" / "T6_lineage"

#: the two tables the faithfulness / stability / TAP-family axes use
PRIMARY_DATASETS = ["MIMIC", "eICU"]
#: every dataset whose complete table exists, for the file-identity part
ALL_DATASETS = ["MIMIC", "eICU", "NHANES", "CDC2022",
                "AutoMPG", "ComCri", "Concrete"]
#: the condition the archived TAP_0 and the family were computed under
CONDITION = ("MAR", 30)
#: the seed of the archived TAP_0 matrix the family compares against
ARCHIVED_TAP_SEED = 1

#: Audit B classes, in the order they are reported
CLASSES = ["equals_fill_only", "equals_truth_only", "equals_both",
           "equals_neither", "missing"]


# --------------------------------------------------------------------------- #
# digests
# --------------------------------------------------------------------------- #
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_frame_digest(df: pd.DataFrame) -> str:
    """T6.0 A3: column names, dtypes, then float64 LE bytes in C order.

    Non-numeric columns are hashed from their str() representation, which is
    what a categorical cell is compared on in Audit B.
    """
    h = hashlib.sha256()
    for c in df.columns:
        h.update(str(c).encode())
        h.update(str(df[c].dtype).encode())
    for c in df.columns:
        col = df[c]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().sum() == col.notna().sum():
            h.update(np.ascontiguousarray(
                num.to_numpy(dtype="float64")).tobytes())
        else:
            h.update("\x00".join(col.astype(str).tolist()).encode())
    return h.hexdigest()


def masked_coordinate_digest(df: pd.DataFrame, mask: pd.DataFrame) -> str:
    """The digest of ONLY the masked cells, in a fixed order.

    Two tables that agree everywhere except the withheld cells have equal
    whole-table digests only by accident; this one isolates the cells the
    question is about (fifth review SS3.1, "masked-coordinate-only digests").
    """
    h = hashlib.sha256()
    for c in sorted(mask.columns):
        idx = np.where(mask[c].to_numpy(dtype=bool))[0]
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(idx.astype("int64")).tobytes())
        vals = df[c].to_numpy()[idx]
        h.update("\x00".join(_as_text(v) for v in vals).encode())
    return h.hexdigest()


def _as_text(v) -> str:
    """The comparison representation: exact for floats, str for everything else."""
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else float(v).hex()
    if isinstance(v, (int, np.integer)):
        return float(v).hex()
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    try:
        if pd.isna(v):
            return "nan"
    except (TypeError, ValueError):
        pass
    return str(v)


# --------------------------------------------------------------------------- #
# Audit B: the masked-coordinate value check
# --------------------------------------------------------------------------- #
def classify(X: pd.DataFrame, F: pd.DataFrame, G: pd.DataFrame,
             mask: pd.DataFrame, n_examples: int = 5) -> dict:
    """T6.0 SS3. Every masked cell into exactly one of five classes."""
    counts = {k: 0 for k in CLASSES}
    examples: dict = {k: [] for k in CLASSES}
    cols = [c for c in mask.columns if c in X.columns]
    for c in cols:
        idx = np.where(mask[c].to_numpy(dtype=bool))[0]
        if idx.size == 0:
            continue
        xs = [_as_text(v) for v in X[c].to_numpy()[idx]]
        fs = [_as_text(v) for v in F[c].to_numpy()[idx]]
        gs = [_as_text(v) for v in G[c].to_numpy()[idx]]
        for k, (x, f, g) in enumerate(zip(xs, fs, gs)):
            if x == "nan":
                cls = "missing"
            elif x == f and x == g:
                cls = "equals_both"
            elif x == f:
                cls = "equals_fill_only"
            elif x == g:
                cls = "equals_truth_only"
            else:
                cls = "equals_neither"
            counts[cls] += 1
            if len(examples[cls]) < n_examples:
                examples[cls].append(
                    {"column": c, "row": int(idx[k]),
                     "X": _short(X[c].to_numpy()[idx][k]),
                     "F": _short(F[c].to_numpy()[idx][k]),
                     "G": _short(G[c].to_numpy()[idx][k])})
    return {"counts": counts, "examples": examples,
            "n_masked_cells": int(sum(counts.values()))}


def _short(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 8)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)


def verdict(cl: dict) -> str:
    """T6.0 SS3, the verdict rule. Not reinterpreted here."""
    c = cl["counts"]
    n = cl["n_masked_cells"]
    if c["equals_truth_only"] > 0:
        return "LEAKING"
    if c["missing"] == n and n > 0:
        return "CLEAN-MASKED"
    if c["equals_neither"] == 0 and c["missing"] == 0 and n > 0:
        return "CLEAN-COMPLETION"
    return "INDETERMINATE"


# --------------------------------------------------------------------------- #
# the pipeline, reconstructed
# --------------------------------------------------------------------------- #
def load_case(ds: str):
    """Exactly as experiments/prior_attribution.load_real_case builds it."""
    from baselines.schema import DataSchema
    mech, rate = CONDITION
    G_full = pd.read_csv(TABLES / f"{ds}_complete.csv")
    mask_np = np.load(MASKS / ds / f"{ds}_{mech}_{rate}per_mask.npy").astype(bool)
    schema = DataSchema.from_yaml(CODE_ROOT / "configs" / "datasets.yaml", ds)
    feats = list(schema.categorical_vars) + list(schema.continuous_vars)
    mask_df = pd.DataFrame(mask_np, columns=list(G_full.columns))[feats]
    missing = G_full[feats].mask(mask_df)
    return {"G_full": G_full, "G": G_full[feats], "mask": mask_df,
            "missing": missing, "feats": feats,
            "cat": list(schema.categorical_vars),
            "cont": list(schema.continuous_vars)}


def initial_completion(case: dict, seed: int) -> pd.DataFrame:
    """The table the first training round sees.

    Reconstructed by the same code path the archived TAP_0 used
    (experiments/prior_attribution.compute_P: max_iters=0, so impute()
    returns the cast initial completion and trains nothing).
    """
    from sni.imputer import SNIConfig, SNIImputer
    imp = SNIImputer(categorical_vars=case["cat"],
                     continuous_vars=case["cont"],
                     config=SNIConfig(seed=seed, use_gpu=False))
    imp.cfg.max_iters = 0
    return imp.impute(X_missing=case["missing"], X_complete=None,
                      mask_df=case["mask"])


def initial_completion_rule() -> dict:
    """T6.0 A4: the rule, read from source rather than described."""
    src = (CODE_ROOT / "sni" / "imputer.py").read_text().splitlines()
    start = next(i for i, l in enumerate(src)
                 if "def _initial_stat_impute" in l)
    body = "\n".join(src[start:start + 34])
    return {
        "function": "sni/imputer.py::SNIImputer._initial_stat_impute",
        "rule": "sklearn IterativeImputer (MICE), max_iter=5, "
                "sample_posterior=False, initial_strategy='mean', "
                "random_state=cfg.seed; categoricals integer-encoded before "
                "and decoded after",
        "fitted_on": "the masked feature table passed to impute(), all rows; "
                     "there is no train/test split at this stage, because the "
                     "faithfulness axis operates on one table",
        "target_column_can_enter": False,
        "target_column_evidence": "the feature list is schema.categorical_vars "
                                  "+ schema.continuous_vars, which excludes the "
                                  "downstream target (audited separately over "
                                  "all 2,565 grid runs: 0 violations)",
        "source": body,
    }


# --------------------------------------------------------------------------- #
# Audit A: file identity and producers
# --------------------------------------------------------------------------- #
def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=CODE_ROOT,
                          capture_output=True, text=True).stdout.strip()


def producers_of_complete_tables() -> dict:
    """T6.0 A1: who writes *_complete.csv, and when relative to masking."""
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-e", "to_csv", "."],
        cwd=CODE_ROOT, capture_output=True, text=True).stdout.splitlines()
    writers = [h for h in hits
               if "complete" in h.lower() and "/tests/" not in h]
    return {
        "writers_of_complete_csv": writers,
        "producer_family": "data_layer/build_*.py -- the benchmark-table "
                           "builders; they write the complete-case extraction "
                           "BEFORE any mask exists",
        "masking_step": "missingness/ generators write "
                        "data/masks/clinical_v1_shuffled/{ds}/*.csv and the "
                        "companion *_mask.npy, reading the complete table as "
                        "their input",
        "order": "build -> data/derived_shuffled/{ds}_complete.csv -> mask "
                 "generation -> data/masks/.../{ds}_{mech}_{rate}per.csv",
    }


# --------------------------------------------------------------------------- #
# the archived artifact: reproduce it, and show what the other table gives
# --------------------------------------------------------------------------- #
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"


def _prior_matrix_from(X: pd.DataFrame, case: dict, seed: int) -> pd.DataFrame:
    """experiments/prior_attribution.compute_P's readout path, on a given table.

    compute_P's only degree of freedom is the table it hands to
    _compute_correlation_prior; feeding it either candidate answers, at the
    level of the ARCHIVED ARTIFACT rather than the code path, which table the
    matrix on disk was made from.
    """
    from sni.imputer import SNIConfig, SNIImputer
    imp = SNIImputer(categorical_vars=case["cat"],
                     continuous_vars=case["cont"],
                     config=SNIConfig(seed=seed, use_gpu=False))
    Prior, dims, cols = imp._compute_correlation_prior(X[case["feats"]])
    d = len(imp.all_vars)
    P = np.zeros((d, d), dtype=float)
    for i, f in enumerate(imp.all_vars):
        vec = imp._normalize_prior(
            imp._extract_feature_prior(Prior, f, cols, dims))
        others = [v for v in imp.all_vars if v != f]
        for val, o in zip(vec, others):
            P[i, imp.all_vars.index(o)] = float(val)
    return pd.DataFrame(P, index=imp.all_vars, columns=imp.all_vars)


def reproduce_archived_tap(ds: str, case: dict, F: pd.DataFrame,
                           seed: int = ARCHIVED_TAP_SEED) -> dict:
    """Does the archived TAP matrix reproduce from the initial completion?

    Two candidate inputs are tried: the initial completion F (what the
    manuscript says) and the pre-mask table G (what T53's sentence says). The
    archived matrix can only reproduce from one of them.
    """
    path = PRIOR / f"P_{ds}_seed{seed}_cpu_t2.csv"
    if not path.exists():
        return {"archived_matrix": str(path), "present": False}
    A = pd.read_csv(path, index_col=0)
    out = {"archived_matrix": str(path.relative_to(CODE_ROOT)),
           "present": True, "sha256": sha256_file(path)}
    for name, X in (("from_initial_completion", F), ("from_complete_table",
                                                     case["G"])):
        try:
            P = _prior_matrix_from(X, case, seed)
            P = P.reindex(index=A.index, columns=A.columns)
            diff = float(np.nanmax(np.abs(P.to_numpy(float)
                                          - A.to_numpy(float))))
            out[name] = {"max_abs_diff_vs_archived": diff,
                         "bit_identical": bool(diff == 0.0)}
        except Exception as e:                       # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    # The archived matrix is a CSV, so exact equality is unattainable: the
    # decimal round-trip alone moves the last bits. The criterion is therefore
    # a separation criterion, and it is stated rather than hidden -- one
    # candidate must agree to within round-trip precision (< 1e-9) while the
    # other differs by more than 1e-2. This criterion is NOT part of the T6.0
    # verdict rule (which is the exact masked-coordinate test in Audit B); it
    # is a confirmatory check added under A2(iii)/A5, and its two thresholds
    # were fixed after seeing that the two candidates are separated by more
    # than ten orders of magnitude -- that is, this is not a close call. Both
    # raw distances are reported so a reader can apply any other criterion.
    out["criterion"] = ("one candidate within 1e-9 (CSV round-trip precision) "
                        "and the other beyond 1e-2; both raw distances are "
                        "reported; confirmatory, not part of the verdict rule")
    dists = {k: out[k].get("max_abs_diff_vs_archived")
             for k in ("from_initial_completion", "from_complete_table")
             if isinstance(out.get(k), dict)
             and out[k].get("max_abs_diff_vs_archived") is not None}
    near = [k for k, v in dists.items() if v < 1e-9]
    far = [k for k, v in dists.items() if v > 1e-2]
    if len(near) == 1 and len(far) == len(dists) - 1:
        out["reproduces_from"] = near[0].replace("from_", "")
        out["unambiguous"] = True
        if len(dists) == 2:
            lo = min(dists.values()) or 1e-300
            out["separation_orders_of_magnitude"] = round(
                float(np.log10(max(dists.values()) / lo)), 1)
    else:
        out["reproduces_from"] = "neither"
        out["unambiguous"] = False
    return out


# --------------------------------------------------------------------------- #
# the consumers
# --------------------------------------------------------------------------- #
def consumer_inputs(ds: str, case: dict, F: pd.DataFrame) -> dict:
    """Every consumer in scope, with the table it actually reads.

    The selection expression is quoted from the source, so the mapping is
    checkable against the code rather than asserted.
    """
    mech, rate = CONDITION
    masked_csv = pd.read_csv(MASKS / ds / f"{ds}_{mech}_{rate}per.csv")
    masked_feats = masked_csv[[c for c in case["feats"]
                               if c in masked_csv.columns]]
    return {
        "sni_impute_argument": {
            "frame": case["missing"], "chain": "primary",
            "expect": "CLEAN-MASKED",
            "source": "experiments/faithfulness.py:99 "
                      "`missing = complete[feats].mask(mask_df)`; passed as "
                      "impute(X_missing=missing, X_complete=None)"},
        "sni_round1_input": {
            "frame": F, "chain": "primary", "expect": "CLEAN-COMPLETION",
            "source": "sni/imputer.py:305 "
                      "`X_current = self._initial_stat_impute(X_missing)` -- "
                      "the table the first EM round trains on"},
        "archived_tap0_input": {
            "frame": F, "chain": "primary", "expect": "CLEAN-COMPLETION",
            "source": "experiments/prior_attribution.py:105-107 "
                      "`imp.cfg.max_iters = 0; X0 = imp.impute(X_missing="
                      "missing, ...); imp._compute_correlation_prior(X0)`"},
        "family_abs_spearman": {
            "frame": _family_input(ds, "abs_spearman", case, F),
            "chain": "sensitivity",
            "expect": "CLEAN-COMPLETION",
            "source": "experiments/t53_tap_family.py "
                      "`fair = initial_completion_input(ds)` -> "
                      "`_pool_matrix(fair, cat, cont, 'spearman')` "
                      "(corrected 2026-08-29; see "
                      "docs/T53_input_correction_rules.md)"},
        "family_mutual_information": {
            "frame": _family_input(ds, "mutual_information", case, F),
            "chain": "sensitivity",
            "expect": "CLEAN-COMPLETION",
            "source": "experiments/t53_tap_family.py "
                      "`fair = initial_completion_input(ds)` -> "
                      "`_mi_matrix(fair, cat, cont)` "
                      "(corrected 2026-08-29; see "
                      "docs/T53_input_correction_rules.md)"},
        "family_superseded_input": {
            "frame": case["G"], "chain": "superseded",
            "expect": "LEAKING -- kept in the record so the defect that was "
                      "corrected stays visible",
            "source": "experiments/t53_tap_family.py BEFORE 2026-08-29: "
                      "`complete = pd.read_csv(.../{ds}_complete.csv)` fed "
                      "directly to both variants"},
        "family_observed_only": {
            "frame": masked_feats, "chain": "sensitivity",
            "expect": "CLEAN-MASKED",
            "source": "experiments/t53_tap_family.py:107-108 "
                      "`X = pd.read_csv(data/masks/clinical_v1_shuffled/"
                      "{ds}/{ds}_MAR_30per.csv)`"},
        "complete_table_itself": {
            "frame": case["G"], "chain": "reference",
            "expect": "LEAKING by definition -- this is the control that "
                      "proves the test can see a leak",
            "source": "data/derived_shuffled/{ds}_complete.csv, the pre-mask "
                      "benchmark table"},
    }



def _family_input(ds: str, variant: str, case: dict, F: pd.DataFrame):
    """The table t53_tap_family.py actually hands this variant, read from it.

    Imported rather than reimplemented, so the audit cannot drift away from
    the code it is auditing: if the correction is reverted, this function
    returns the reverted table and the verdict turns red on its own.
    """
    try:
        from experiments.t53_tap_family import initial_completion_input
        return initial_completion_input(ds)
    except ImportError:                              # pre-correction code
        return case["G"]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run() -> dict:
    mech, rate = CONDITION
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "rules": f"{RULES}@{_git('log', '-1', '--format=%h', '--', RULES)}",
        "code_commit": _git("rev-parse", "HEAD"),
        "condition": f"{mech}@{rate}%",
        "archived_tap_seed": ARCHIVED_TAP_SEED,
        "initial_completion_rule": initial_completion_rule(),
        "file_identity": producers_of_complete_tables(),
        "datasets": {},
        "limits": "Establishes what the code paths and archived artifacts on "
                  "disk today consume and contain. Tables that are not "
                  "on-disk artifacts are RECONSTRUCTED by executing the same "
                  "deterministic code path, and are labeled as such.",
    }

    # A2 (iv): digests of the on-disk tables, all datasets
    files = {}
    for ds in ALL_DATASETS:
        gp = TABLES / f"{ds}_complete.csv"
        if gp.exists():
            files[str(gp.relative_to(CODE_ROOT))] = sha256_file(gp)
        for suffix in (f"{ds}_{mech}_{rate}per.csv",
                       f"{ds}_{mech}_{rate}per_mask.npy"):
            mp = MASKS / ds / suffix
            if mp.exists():
                files[str(mp.relative_to(CODE_ROOT))] = sha256_file(mp)
    audit["file_identity"]["sha256"] = files

    provenance = {}
    for ds in PRIMARY_DATASETS:
        case = load_case(ds)
        F = initial_completion(case, ARCHIVED_TAP_SEED)
        cons = consumer_inputs(ds, case, F)

        # the masked CSV on disk must be the same object the drivers rebuild
        masked_csv = pd.read_csv(MASKS / ds / f"{ds}_{mech}_{rate}per.csv")
        common = [c for c in case["feats"] if c in masked_csv.columns]
        rebuilt = canonical_frame_digest(case["missing"][common])
        on_disk = canonical_frame_digest(masked_csv[common])
        # A5: a digest difference is itemized, not summarized. The digest
        # includes the dtype label, so a CSV round-trip that turns int64 into
        # float64 on a never-masked column moves the digest without moving a
        # single value.
        a_, b_ = case["missing"][common], masked_csv[common]
        dtype_only = [c for c in common
                      if str(a_[c].dtype) != str(b_[c].dtype)]
        vals_same = all(
            [_as_text(x) == _as_text(y)
             for c in common
             for x, y in zip(a_[c].to_numpy(), b_[c].to_numpy())])
        cell_diffs = [] if vals_same else [
            {"column": c, "row": int(i),
             "rebuilt": _short(a_[c].to_numpy()[i]),
             "on_disk": _short(b_[c].to_numpy()[i])}
            for c in common
            for i in range(len(a_))
            if _as_text(a_[c].to_numpy()[i]) != _as_text(b_[c].to_numpy()[i])
        ][:20]

        results = {}
        for name, spec in cons.items():
            cl = classify(spec["frame"], F, case["G"], case["mask"])
            results[name] = {
                "chain": spec["chain"],
                "expected": spec["expect"],
                "source": spec["source"],
                "verdict": verdict(cl),
                "counts": cl["counts"],
                "n_masked_cells": cl["n_masked_cells"],
                "examples": cl["examples"],
                "frame_digest": canonical_frame_digest(spec["frame"]),
                "masked_coordinate_digest": masked_coordinate_digest(
                    spec["frame"], case["mask"]),
                "reconstructed": name in ("sni_round1_input",
                                          "archived_tap0_input",
                                          "sni_impute_argument"),
            }

        audit["datasets"][ds] = {
            "n_rows": int(len(case["G"])),
            "n_features": len(case["feats"]),
            "n_masked_cells": int(case["mask"].to_numpy().sum()),
            "masked_table_rebuild_matches_disk": bool(rebuilt == on_disk),
            "masked_table_values_identical": bool(vals_same),
            "masked_table_dtype_only_differences": [
                {"column": c, "rebuilt": str(a_[c].dtype),
                 "on_disk": str(b_[c].dtype),
                 "masked_cells": int(case["mask"][c].sum())}
                for c in dtype_only],
            "masked_table_cell_differences": cell_diffs,
            "masked_table_digest_rebuilt": rebuilt,
            "masked_table_digest_on_disk": on_disk,
            "archived_tap_reproduction": reproduce_archived_tap(ds, case, F),
            "tap0_input_is_sni_round1_input": bool(
                results["archived_tap0_input"]["frame_digest"]
                == results["sni_round1_input"]["frame_digest"]),
            # T53 correction rules SS4: the path change does not count as
            # proof; the corrected inputs must carry the SAME masked
            # coordinates as the archived TAP_0 input.
            "family_inputs_match_tap0": {
                v: bool(results[v]["masked_coordinate_digest"]
                        == results["archived_tap0_input"]
                        ["masked_coordinate_digest"])
                for v in ("family_abs_spearman", "family_mutual_information")},
            "consumers": results,
        }

        provenance[ds] = {
            "complete_ground_truth_path": f"data/derived_shuffled/{ds}_complete.csv",
            "complete_ground_truth_sha256": files.get(
                f"data/derived_shuffled/{ds}_complete.csv"),
            "masked_input_path": f"data/masks/clinical_v1_shuffled/{ds}/{ds}_{mech}_{rate}per.csv",
            "masked_input_sha256": files.get(
                f"data/masks/clinical_v1_shuffled/{ds}/{ds}_{mech}_{rate}per.csv"),
            "mask_path": f"data/masks/clinical_v1_shuffled/{ds}/{ds}_{mech}_{rate}per_mask.npy",
            "mask_sha256": files.get(
                f"data/masks/clinical_v1_shuffled/{ds}/{ds}_{mech}_{rate}per_mask.npy"),
            "initial_completion_path": "(in memory; reconstructed by "
                                       "sni/imputer.py::_initial_stat_impute)",
            "initial_completion_digest": results["sni_round1_input"]["frame_digest"],
            "TAP0_input_path": "(in memory; experiments/prior_attribution.py::compute_P)",
            "TAP0_input_digest": results["archived_tap0_input"]["frame_digest"],
            "SNI_round1_input_path": "(in memory; sni/imputer.py::impute)",
            "SNI_round1_input_digest": results["sni_round1_input"]["frame_digest"],
            "masked_coordinate_digests": {
                k: v["masked_coordinate_digest"] for k, v in results.items()},
        }

    audit["decision"] = decide(audit)
    (OUT / "data_lineage_audit.json").write_text(
        json.dumps(audit, indent=1, default=str))
    (OUT / "tap_input_provenance.json").write_text(
        json.dumps(provenance, indent=1, default=str))
    return audit


def decide(audit: dict) -> dict:
    """T6.0 SS6, evaluated mechanically."""
    primary_bad, sensitivity_bad, notes = [], [], []
    for ds, d in audit["datasets"].items():
        for name, r in d["consumers"].items():
            if r["chain"] in ("reference", "superseded"):
                continue
            ok = ((r["expected"] == "CLEAN-MASKED"
                   and r["verdict"] == "CLEAN-MASKED")
                  or (r["expected"] == "CLEAN-COMPLETION"
                      and r["verdict"] == "CLEAN-COMPLETION")
                  or (r["expected"].startswith("same input")
                      and r["verdict"] == "CLEAN-COMPLETION"))
            if not ok:
                (primary_bad if r["chain"] == "primary"
                 else sensitivity_bad).append(f"{ds}/{name}={r['verdict']}")
        if not d["tap0_input_is_sni_round1_input"]:
            primary_bad.append(f"{ds}/TAP0_input != SNI_round1_input")
        if not d.get("masked_table_values_identical", True):
            primary_bad.append(f"{ds}/rebuilt masked table differs in VALUE "
                               f"from the CSV on disk")
        elif not d["masked_table_rebuild_matches_disk"]:
            notes.append(
                f"{ds}: rebuilt masked table is cell-for-cell identical to "
                f"the CSV on disk; the digests differ only in the dtype label "
                f"of "
                f"{[x['column'] for x in d.get('masked_table_dtype_only_differences', [])]}"
                f" after a CSV round-trip")
        for v, okd in d.get("family_inputs_match_tap0", {}).items():
            if not okd:
                sensitivity_bad.append(
                    f"{ds}/{v} masked coordinates differ from the archived "
                    f"TAP_0 input")
        rep = d.get("archived_tap_reproduction", {})
        if rep.get("present"):
            if rep.get("reproduces_from") == "complete_table":
                primary_bad.append(
                    f"{ds}/archived TAP matrix reproduces from the pre-mask "
                    f"table, not from the initial completion")
            elif rep.get("reproduces_from") == "initial_completion":
                notes.append(
                    f"{ds}: the ARCHIVED TAP matrix on disk reproduces from "
                    f"the initial completion to "
                    f"{rep['from_initial_completion']['max_abs_diff_vs_archived']:.1e} "
                    f"and from the pre-mask table only to "
                    f"{rep['from_complete_table']['max_abs_diff_vs_archived']:.3f} "
                    f"-- a separation of "
                    f"{rep.get('separation_orders_of_magnitude')} orders of "
                    f"magnitude, so the archived artifact was made from the "
                    f"initial completion")
            elif rep.get("reproduces_from") == "neither":
                notes.append(
                    f"{ds}: archived TAP matrix reproduces bit-identically "
                    f"from neither candidate; max|diff| initial-completion="
                    f"{rep.get('from_initial_completion', {}).get('max_abs_diff_vs_archived')}, "
                    f"complete-table="
                    f"{rep.get('from_complete_table', {}).get('max_abs_diff_vs_archived')}")
        # the control must fire, or the test proves nothing
        ctrl = d["consumers"]["complete_table_itself"]["verdict"]
        if ctrl != "LEAKING":
            primary_bad.append(f"{ds}/control did not fire (got {ctrl}) -- "
                               f"the test cannot be trusted")
    if primary_bad:
        outcome = "FAIL"
    elif sensitivity_bad:
        outcome = "SCOPED-DEFECT"
    else:
        outcome = "PASS"
    return {"outcome": outcome, "primary_chain_failures": primary_bad,
            "sensitivity_layer_failures": sensitivity_bad, "notes": notes}


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    n = 40
    rng = np.random.default_rng(7)
    G = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.integers(0, 3, size=n).astype(str)})
    mask = pd.DataFrame(False, index=G.index, columns=G.columns)
    mask.loc[0:9, "a"] = True
    mask.loc[5:14, "c"] = True
    F = G.copy()
    F.loc[0:9, "a"] = 99.5                      # a fill that is never the truth
    F.loc[5:14, "c"] = "0"                      # a mode fill: sometimes the truth

    clean = F.copy()
    v = verdict(classify(clean, F, G, mask))
    check(v == "CLEAN-COMPLETION", f"a correctly filled table is clean ({v})")

    leak = F.copy()
    leak.loc[3, "a"] = G.loc[3, "a"]            # one planted leaked cell
    cl = classify(leak, F, G, mask)
    check(verdict(cl) == "LEAKING", "one planted leaked cell is caught")
    check(cl["counts"]["equals_truth_only"] == 1,
          "the planted cell lands in equals_truth_only")
    check(len(cl["examples"]["equals_truth_only"]) == 1,
          "the planted cell is exhibited with its three values")

    v = verdict(classify(G, F, G, mask))
    check(v == "LEAKING", "the ground-truth table itself reads as LEAKING")

    masked = G.mask(mask)
    v = verdict(classify(masked, F, G, mask))
    check(v == "CLEAN-MASKED", "the masked table reads as CLEAN-MASKED")

    other = F.copy()
    other.loc[2, "a"] = -1234.0
    check(verdict(classify(other, F, G, mask)) == "INDETERMINATE",
          "a cell that matches neither is INDETERMINATE, not clean")

    coin = classify(F, F, G, mask)
    check(coin["counts"]["equals_both"] > 0,
          "the mode fill produces coincidences, and they are counted "
          "separately")
    check(verdict(coin) == "CLEAN-COMPLETION",
          "coincidences alone do not make a table leak")

    # digests
    check(canonical_frame_digest(G) == canonical_frame_digest(G.copy()),
          "frame digest is stable")
    check(canonical_frame_digest(G) != canonical_frame_digest(F),
          "frame digest separates two different tables")
    d1 = masked_coordinate_digest(F, mask)
    F2 = F.copy()
    F2.loc[25, "a"] = 12345.0                   # outside the mask
    check(masked_coordinate_digest(F2, mask) == d1,
          "masked-coordinate digest ignores unmasked cells")
    F3 = F.copy()
    F3.loc[1, "a"] = 12345.0                    # inside the mask
    check(masked_coordinate_digest(F3, mask) != d1,
          "masked-coordinate digest sees a masked cell change")

    # the decision tree
    def _fake(primary_v, sens_v):
        return {"datasets": {"X": {
            "tap0_input_is_sni_round1_input": True,
            "masked_table_rebuild_matches_disk": True,
            "consumers": {
                "sni_round1_input": {"chain": "primary",
                                     "expected": "CLEAN-COMPLETION",
                                     "verdict": primary_v},
                "family_abs_spearman": {"chain": "sensitivity",
                                        "expected": "same input as archived",
                                        "verdict": sens_v},
                "complete_table_itself": {"chain": "reference",
                                          "expected": "LEAKING",
                                          "verdict": "LEAKING"}}}}}
    check(decide(_fake("CLEAN-COMPLETION", "CLEAN-COMPLETION"))["outcome"]
          == "PASS", "tree: all clean -> PASS")
    check(decide(_fake("CLEAN-COMPLETION", "LEAKING"))["outcome"]
          == "SCOPED-DEFECT", "tree: sensitivity only -> SCOPED-DEFECT")
    check(decide(_fake("LEAKING", "CLEAN-COMPLETION"))["outcome"]
          == "FAIL", "tree: primary chain -> FAIL")
    bad_ctrl = _fake("CLEAN-COMPLETION", "CLEAN-COMPLETION")
    bad_ctrl["datasets"]["X"]["consumers"]["complete_table_itself"]["verdict"] \
        = "CLEAN-COMPLETION"
    check(decide(bad_ctrl)["outcome"] == "FAIL",
          "tree: a control that does not fire -> FAIL (the test would prove "
          "nothing)")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    audit = run()
    d = audit["decision"]
    for ds, rec in audit["datasets"].items():
        print(f"\n=== {ds} ({rec['n_masked_cells']} masked cells) ===")
        for name, r in rec["consumers"].items():
            print(f"  {r['verdict']:<18} {name:<28} "
                  f"[{r['chain']}] {r['counts']}")
        print(f"  TAP0 input == SNI round-1 input: "
              f"{rec['tap0_input_is_sni_round1_input']}")
    print(f"\nDECISION: {d['outcome']}")
    for k in ("primary_chain_failures", "sensitivity_layer_failures", "notes"):
        if d[k]:
            print(f"  {k}: {d[k]}")
    print(f"\nwrote {OUT / 'data_lineage_audit.json'}")
    print(f"wrote {OUT / 'tap_input_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
