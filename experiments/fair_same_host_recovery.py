"""The same-host recovery comparison, with both sides from one trained host.

T6.1 addendum 2026-08-29d SS2, from the sixth review's first finding. The
manuscript's carrier-controlled recovery claim -- the same-host probe leads the
attention matrix in all fifteen paired cells, T = +0.154 -- pairs an ablation
whose error signal reads the values withheld from the imputer against an
attention matrix that has no such access. Removing the oracle from the probe
while leaving D on a different host would swap one confound for another, so
this trains ONE host per cell and takes both objects from it:

    D                the host's own attention matrix, row-normalized, exactly
                     as t4f_perm_on_sni.run_synth computes it
    PERM_fair_oracle    the ablation with the error signal from the withheld
                     values -- a control, so the oracle's contribution is
                     separable from the host draw
    PERM_fair_noOracle  the same ablation with the error signal from the host's
                     own completed table

The pair is within-host, so the host draw cancels in the comparison. Scoring is
t4f_perm_on_sni.stage_score's algorithm verbatim; the estimand is the axis's
own (mean of seed-level medians, seed-only bootstrap, exact sign enumeration
at the same five-seed floor).

Resumable: a cell whose three matrices are on disk is skipped.

    env PYTHONHASHSEED=2025 python experiments/fair_same_host_recovery.py
    env PYTHONHASHSEED=2025 python experiments/fair_same_host_recovery.py --score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

PILOT = CODE_ROOT / "results" / "T2.5_pilot"
PRIOR = CODE_ROOT / "results" / "T2g_prior_attribution"
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
OUT = CODE_ROOT / "results" / "T6_symmetry"
REGIMES = ["linear_gaussian", "nonlinear_mixed", "interaction_xor"]
SEEDS = [2025, 2026, 2027, 2028, 2029]


def _contained(p: Path) -> Path:
    rp = Path(p).resolve()
    if OUT.resolve() not in rp.parents:
        raise RuntimeError(f"refusing to write outside {OUT}: {rp}")
    return rp


def _paths(tag: str) -> dict:
    return {"D": OUT / f"D_FAIR_{tag}.csv",
            "or": OUT / f"PERM_fair_oracle_{tag}.csv",
            "no": OUT / f"PERM_fair_noOracle_{tag}.csv",
            "meta": OUT / f"meta_fair_{tag}.json"}


def train_cells(regimes=None, seeds=None) -> dict:
    import yaml
    from common import determinism
    from pilot_r21 import load_cell, row_normalise
    from sni.imputer import SNIConfig, SNIImputer
    from no_oracle_recompute import ablate

    regimes = regimes or REGIMES
    seeds = seeds or SEEDS
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {}
    for regime in regimes:
        for seed in seeds:
            tag = f"{regime}_s{seed}"
            f = _paths(tag)
            if all(f[k].exists() for k in ("D", "or", "no")):
                print(f"[cached] {tag}", flush=True)
                continue
            t0 = time.time()
            complete, missing, mask, G, cat, cont, _ = load_cell(regime, seed)
            cols = list(complete.columns)
            proto = yaml.safe_load(
                (CODE_ROOT / "configs" / "training_protocol.yaml").read_text()
            )["protocol"]
            determinism.apply("deterministic", seed=seed)
            imp = SNIImputer(categorical_vars=cat, continuous_vars=cont,
                             config=SNIConfig(seed=seed, use_gpu=False))
            imp.cfg.epochs = int(proto["epochs"]["SNI"])
            imp.cfg.early_stopping_patience = imp.cfg.epochs + 1
            X_final = imp.impute(X_missing=missing[imp.all_vars],
                                 X_complete=None, mask_df=mask[imp.all_vars])
            train_s = time.time() - t0

            D = row_normalise(imp.compute_dependency_matrix().reindex(
                index=cols, columns=cols).fillna(0.0))
            mask_df = mask[imp.all_vars].astype(bool)
            A_or = ablate(imp, X_final, mask_df, complete, imp.all_vars,
                          seed, set(cat)).reindex(index=cols, columns=cols)
            A_no = ablate(imp, X_final, mask_df, X_final, imp.all_vars,
                          seed, set(cat)).reindex(index=cols, columns=cols)
            D.to_csv(_contained(f["D"]))
            A_or.to_csv(_contained(f["or"]))
            A_no.to_csv(_contained(f["no"]))
            cells[tag] = {"train_sec": round(train_s, 1),
                          "total_sec": round(time.time() - t0, 1)}
            _contained(f["meta"]).write_text(json.dumps(cells[tag], indent=1))
            print(f"[ok] {tag} train={train_s:.0f}s "
                  f"total={time.time()-t0:.0f}s", flush=True)
    return cells


# The retired rule's arithmetic used to live here as _verdict(): a Wilcoxon
# over fifteen regime-by-seed cells treated as fifteen independent pairs,
# plus the conditions that produced SAME_HOST_POSTHOC_WINS.
#
# It is deleted, not disabled. The ninth round stopped applying it to this
# study's data and kept running it on the frozen T4F record as an "integrity
# check"; the tenth review's ruling is that **re-executing a rule known to be
# invalid is resurrection whatever data it runs on** -- and it is right.
# Verifying that a frozen record is intact does not require re-deriving it:
# a digest does that, and a digest cannot accidentally become a current
# result. See _frozen_t4f_integrity() below.


def _frozen_t4f_integrity() -> dict:
    """Is the frozen T4F record intact? -- answered without re-deriving it.

    Three things, none of which runs the retired rule (tenth review P0-1):

      digest  the SHA-256 of the two frozen files, so any edit shows up;
      schema  the keys the record must carry, so a truncated or reshaped
              file is not mistaken for an intact one;
      path    where the record lives, stated, so "the frozen record" is a
              file a reader can open rather than a phrase.

    The label the retired rule once produced is NOT read here and NOT copied
    into this summary. It lives in evidence/AUDIT_HISTORY.json, which is
    marked HISTORICAL - NOT CANONICAL, and nothing current reads that either.
    """
    import hashlib
    out = {"method": "digest + schema + declared path; the retired rule's "
                     "arithmetic is NOT re-executed (tenth review P0-1)",
           "label_lives_in": "evidence/AUDIT_HISTORY.json "
                             "(HISTORICAL - NOT CANONICAL)",
           "current_inferential_source":
               "probe_vs_D_same_host_no_oracle.exact_sign_enumeration "
               "(5 seed blocks, two-sided exact p = 0.0625 at its attainable "
               "floor; classification INDET)",
           "files": {}}
    WANT = {"t4f_verdict.json": ("primary", "verdict"),
            "t4f_sixway_cells.csv": ("regime", "seed", "method", "auroc")}
    for name, keys in WANT.items():
        f = T4F / name
        rec = {"path": f"results/T4_perm_on_sni/{name}",
               "present": f.exists()}
        if f.exists():
            raw = f.read_bytes()
            rec["sha256"] = hashlib.sha256(raw).hexdigest()
            if name.endswith(".json"):
                top = set(json.loads(raw))
                rec["schema_keys_present"] = sorted(set(keys) & top)
                rec["schema_ok"] = set(keys) <= top
            else:
                cols = raw.decode().splitlines()[0].split(",")
                rec["schema_keys_present"] = sorted(set(keys) & set(cols))
                rec["schema_ok"] = set(keys) <= set(cols)
        out["files"][name] = rec
    out["intact"] = all(r.get("present") and r.get("schema_ok")
                        for r in out["files"].values())
    return out




def score() -> dict:
    """stage_score's algorithm verbatim, with D and the probe from ONE host."""
    from pilot_r21 import load_cell, measured_rows, score as _score
    from t4f_perm_on_sni import PILOT_METHODS
    from t51_cluster_stats import sign_flip_exact, seed_boot_ci_T

    rows = []
    for regime in REGIMES:
        for seed in SEEDS:
            tag = f"{regime}_s{seed}"
            f = _paths(tag)
            missing = [k for k in ("D", "or", "no") if not f[k].exists()]
            if missing:
                raise FileNotFoundError(
                    f"{tag}: {missing} absent -- the fair pair is incomplete, "
                    f"and a partial run must not be scored as if it were the "
                    f"comparison (addendum 2026-08-29d SS2).")
            complete, _m, _k, G, _c, _n, _ = load_cell(regime, seed)
            cols = list(complete.columns)

            def rd(p):
                return pd.read_csv(p, index_col=0).reindex(
                    index=cols, columns=cols).fillna(0.0)

            mats = {m: rd(PILOT / f"D_{regime}_s{seed}_{m}.csv")
                    for m in PILOT_METHODS}
            mats["P-alone"] = rd(PRIOR / f"P_synth_{regime}_s{seed}.csv")
            mats["SNI-D-fairhost"] = rd(f["D"])
            mats["Permutation-on-SNI-fair-oracle"] = rd(f["or"])
            mats["Permutation-on-SNI-fair-noOracle"] = rd(f["no"])
            common = np.ones(len(cols), dtype=bool)
            for M in mats.values():
                common &= measured_rows(M)
            for name, M in mats.items():
                rows.append({"regime": regime, "seed": seed, "method": name,
                             "n_rows_common": int(common.sum()),
                             **_score(M, G, keep=common)})
    df = pd.DataFrame(rows)
    df.to_csv(_contained(OUT / "fair_same_host_recovery_cells.csv"), index=False)

    piv = df.pivot_table(index=["regime", "seed"], columns="method",
                         values="auroc")
    out = {"rules": "docs/T61_information_symmetry_rules.md addendum "
                    "2026-08-29d SS2",
           "n_cells": len(piv),
           "means": piv.mean().round(6).to_dict()}
    for label, probe in (("no_oracle", "Permutation-on-SNI-fair-noOracle"),
                         ("oracle_control", "Permutation-on-SNI-fair-oracle")):
        d = (piv[probe] - piv["SNI-D-fairhost"]).dropna()
        by_seed = {int(s): d.xs(s, level="seed").to_numpy(float) for s in SEEDS}
        T = float(np.mean([float(np.median(v)) for v in by_seed.values()]))
        out[f"probe_vs_D_same_host_{label}"] = {
            "T_mean_of_seed_medians": T,
            "ci95_T_seedboot": seed_boot_ci_T(by_seed),
            "exact_sign_enumeration": sign_flip_exact(by_seed),
            "per_regime_median": {r: float(np.median(d.xs(r, level="regime")))
                                  for r in REGIMES},
            "cells_favouring_probe": int((d > 0).sum()),
            "cells_total": int(len(d)),
        }
    a = out["probe_vs_D_same_host_oracle_control"]["T_mean_of_seed_medians"]
    b = out["probe_vs_D_same_host_no_oracle"]["T_mean_of_seed_medians"]
    out["oracle_contribution_T"] = round(a - b, 6)
    # the superseded number is READ from what the manuscript published, not
    # typed here: a typed number cannot go stale, and that is the problem
    tf = json.loads(
        (CODE_ROOT / "results" / "T5_stats" / "t_final.json").read_text()
    )["recovery"]["probe_vs_D_retrained"]
    out["archived_superseded"] = {
        "source": "results/T5_stats/t_final.json recovery.probe_vs_D_retrained",
        "T": tf["T"], "ci95_T": tf["ci95_T"], "p_exact": tf["p_exact"],
        "note": "oracle caliber, and paired with the T4F-RETRAINED host's D, "
                "not the pilot host's; superseded by the within-host pair "
                "above"}

    # Whether the fresh run reproduces the pilot D archive and the T61
    # recomputation archive is a measurement, not an
    # assumption. The addendum could not settle it by reading -- the T6.1
    # recompute saved no weights and no D -- so it is settled here, by
    # comparing what this run produced against what the archive holds.
    rep = {"max_abs_D_fair_minus_pilot_D": 0.0,
           "max_abs_probe_fair_minus_T61_recompute": 0.0, "cells": 0}
    for regime in REGIMES:
        for seed in SEEDS:
            tag = f"{regime}_s{seed}"
            f = _paths(tag)
            cols = list(pd.read_csv(f["D"], index_col=0).columns)

            def rd2(p_):
                return pd.read_csv(p_, index_col=0).reindex(
                    index=cols, columns=cols).to_numpy(float)

            rep["max_abs_D_fair_minus_pilot_D"] = max(
                rep["max_abs_D_fair_minus_pilot_D"],
                float(np.nanmax(np.abs(
                    rd2(f["D"])
                    - rd2(PILOT / f"D_{regime}_s{seed}_SNI-D.csv")))))
            for k, arch in (("no", f"PERM_noOracle_{tag}.csv"),
                            ("or", f"PERM_oracle_{tag}.csv")):
                if (OUT / arch).exists():
                    rep["max_abs_probe_fair_minus_T61_recompute"] = max(
                        rep["max_abs_probe_fair_minus_T61_recompute"],
                        float(np.nanmax(np.abs(rd2(f[k])
                                               - rd2(OUT / arch)))))
            rep["cells"] += 1
    # Eighth review P0-6. What is compared here is two READOUTS -- the
    # attention matrix and the permutation matrix -- cell by cell. No model
    # weights, optimizer state or full prediction vector is compared, because
    # the archive carries no model-state digest to compare against. The old
    # key said "host_is_bitwise_the_archived_host", which claims more than the
    # measurement can support, so it is gone rather than renamed-and-kept.
    rep["relevant_readouts_bitwise_match_archived"] = (
        rep["max_abs_D_fair_minus_pilot_D"] == 0.0
        and rep["max_abs_probe_fair_minus_T61_recompute"] == 0.0)
    rep["scope"] = ("D and the permutation matrix, per cell, on all "
                    f"{rep['cells']} cells. Model weights, optimizer state and "
                    "full prediction vectors are NOT compared: the archive "
                    "holds no model-state digest. This establishes "
                    "readout-level reproduction for the quantities used here, "
                    "not bitwise identity of the full trained host.")
    out["host_reproduction"] = rep

    # ---- the frozen T4F record: integrity by DIGEST, never by re-running -- #
    # Tenth review P0-1, and the last correction this rule needs.
    #
    # The ninth round stopped applying the retired rule to this study's data
    # and kept re-executing it on the frozen T4F record, calling that an
    # integrity check. It is not: re-deriving a result with a rule known to be
    # invalid is running that rule, and a number so produced is one careless
    # edit away from being read as current. Checking that a frozen file is
    # intact needs a digest, and a digest cannot become a result by accident.
    out["frozen_t4f_integrity"] = _frozen_t4f_integrity()
    # The archived reading keeps its T/CI/p, which are legitimate history;
    # the verdict STRING is not carried forward in any form (ninth review
    # P0-1: the selftest below now searches the whole document for it).
    out["archived_superseded"].pop("verdict", None)
    _contained(OUT / "fair_same_host_recovery.json").write_text(
        json.dumps(out, indent=1))
    return out


def _selftest() -> int:
    ok, skipped = True, []

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    def skip(m, why):
        skipped.append(m)
        print(f"[SKIP] {m} -- {why}")

    # --- checks that need the training modules (repository only) ----------
    try:
        from pilot_r21 import row_normalise, score, measured_rows
        from no_oracle_recompute import ablate
        from t4f_perm_on_sni import PILOT_METHODS
    except ModuleNotFoundError as e:
        skip("scorer, row-normaliser, ablation and PILOT_METHODS are the "
             "archived ones",
             f"{e.name} is not in this tree (expected inside a review "
             f"package: the training modules do not ship, and the checks "
             f"below need none of them)")
    else:
        chk(all(callable(x)
                for x in (row_normalise, score, measured_rows, ablate)),
            "scorer, row-normaliser and ablation are the archived ones")
        chk("SNI-D" in PILOT_METHODS,
            f"PILOT_METHODS from t4f ({PILOT_METHODS})")

    try:
        _contained(T4F / "x.csv")
        chk(False, "the write guard refuses a path outside T6_symmetry")
    except RuntimeError:
        chk(True, "the write guard refuses a path outside T6_symmetry")
    src = Path(__file__).read_text()
    chk("D_FAIR_" in src and "PERM_fair_noOracle_" in src,
        "D and the probe are written per cell from the same run")
    chk("raise FileNotFoundError" in src,
        "scoring refuses an incomplete run rather than reporting a partial one")
    body = src.split("def _selftest")[0]   # the check's own text is not code
    chk("t_final.json" in body and "0.1542" not in body,
        "the superseded number is read from t_final.json, not typed")
    art = next((c for c in (
        OUT / "fair_same_host_recovery.json",
        # package layout: code/experiments/<this> and the artifact under
        # gate_inputs/code_SNI/results/, or beside it in evidence/
        CODE_ROOT.parent / "gate_inputs" / "code_SNI" / "results"
        / "T6_symmetry" / "fair_same_host_recovery.json",
        CODE_ROOT.parent / "evidence" / "fair_same_host_recovery.json",
    ) if c.exists()), OUT / "fair_same_host_recovery.json")
    if art.exists():
        d = json.loads(art.read_text())
        rp = d.get("host_reproduction", {})
        chk("relevant_readouts_bitwise_match_archived" in rp
            and "host_is_bitwise_the_archived_host" not in rp
            and "model-state digest" in rp.get("scope", ""),
            "readout-level reproduction is what is emitted, and the old "
            "full-host claim is gone (eighth review P0-6)")
        blob = json.dumps(d)
        from audit_history import RETIRED_LABEL
        banned = ["verdict_under_symmetry", "verdict_oracle_control",
                  RETIRED_LABEL]
        found = [b for b in banned if b in blob]
        chk(not found,
            f"the retired rule's fields and verdict appear NOWHERE in the "
            f"summary -- no namespace exempted (found={found})")
        fz = d.get("frozen_t4f_integrity", {})
        chk(fz.get("intact") is True and "NOT re-executed" in fz.get("method", ""),
            "the frozen record's integrity is established by digest and "
            "schema; the retired rule is not executed on any data")
        chk("retired_audit_history" not in d,
            "the retired namespace is gone: nothing in this summary is a "
            "product of the retired rule")
    else:
        chk(False, f"summary artifact absent (looked beside the script and "
                   f"in a package's gate_inputs/ and evidence/); run --score "
                   f"first, or run this from an unpacked review package")
    if skipped:
        print(f"[NOTE] {len(skipped)} check(s) skipped for want of the "
              f"training modules; every artifact check above ran.")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--regimes", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.score:
        r = score()
        for k in ("probe_vs_D_same_host_no_oracle",
                  "probe_vs_D_same_host_oracle_control"):
            v = r[k]
            print(f"{k}: T={v['T_mean_of_seed_medians']:+.4f} "
                  f"CI={v['ci95_T_seedboot']} "
                  f"p={v['exact_sign_enumeration'].get('p_two_sided')} "
                  f"cells {v['cells_favouring_probe']}/{v['cells_total']}")
        print(f"oracle contribution to T: {r['oracle_contribution_T']:+.4f}")
        return 0
    train_cells(a.regimes, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
