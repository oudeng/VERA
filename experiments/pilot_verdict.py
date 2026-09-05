"""Compute the T2.5 go/no-go mechanically from the rule committed beforehand.

The rule is `docs/T25_pilot_decision_rule.md`, committed at 41c522d before the
first pilot run. This file implements those conditions and no others, and prints
every input so the arithmetic can be checked without rerunning anything.

Same discipline as `tests/t2d1_verdict.py`. The reason it matters more here is
stated in P2e §11: the pilot is the step most exposed to expectation, and a
verdict reached by eye would be worth nothing whichever way it came out.

    env PYTHONHASHSEED=2025 python experiments/pilot_verdict.py
"""

from __future__ import annotations

import glob
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT = CODE_ROOT / "results" / "T2.5_pilot"

SNI = "SNI-D"
POST_HOC = ["MissForest-importance", "SHAP-on-MissForest",
            "Permutation-on-MissForest"]

GO_MARGIN = -0.02          # Δ at or above this is GO
ALPHA = 0.05
COST_FRACTION = 0.50       # CONDITIONAL GO needs SNI total <= this x the best's
STABILITY_FLOOR = 0.50     # independent NO-GO below this median Spearman


def cells(tag: str = "pilot_full") -> pd.DataFrame:
    p = OUT / f"{tag}_cells.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def cross_seed_spearman(method: str) -> tuple:
    """Median pairwise Spearman of one method's D across seeds, within regime."""
    from scipy.stats import spearmanr

    rows = []
    for f in glob.glob(str(OUT / f"D_*_{method}.csv")):
        stem = Path(f).stem[2:]                       # strip "D_"
        regime, rest = stem.rsplit("_s", 1)
        seed = rest.split("_")[0]
        rows.append((regime, seed, pd.read_csv(f, index_col=0)))
    per_regime, pairs = {}, []
    for regime in sorted({r for r, _, _ in rows}):
        mats = {s: M for r, s, M in rows if r == regime}
        rho = []
        for a, b in combinations(sorted(mats), 2):
            A, B = mats[a].to_numpy(float), mats[b].to_numpy(float)
            off = ~np.eye(len(A), dtype=bool)
            r = spearmanr(A[off], B[off]).statistic
            rho.append(r)
            pairs.append((regime, a, b, r))
        if rho:
            per_regime[regime] = float(np.median(rho))
    allr = [r for _, _, _, r in pairs]
    return (float(np.median(allr)) if allr else float("nan")), per_regime, pairs


def main() -> int:
    print("T2.5 verdict — computed from docs/T25_pilot_decision_rule.md "
          "(committed 41c522d, before any run)\n")
    df = cells()
    if df.empty:
        print("no pilot cells yet -> results/T2.5_pilot/pilot_full_cells.csv "
              "missing", file=sys.stderr)
        return 2

    n_cells = df.groupby("method").size().min()
    expected = df.regime.nunique() * df.seed.nunique()
    print(f"cells present: {int(n_cells)} per method "
          f"({df.regime.nunique()} regimes x {df.seed.nunique()} seeds "
          f"= {expected} expected)")
    if n_cells < expected:
        print("  PARTIAL — the verdict below is provisional and must not be "
              "quoted as final.\n")

    piv = df.pivot_table(index=["regime", "seed"], columns="method",
                         values="auroc")
    means = piv.mean().sort_values(ascending=False)
    print("\nmean AUROC over available cells:")
    for m, v in means.items():
        print(f"  {m:<28} {v:.4f}" + ("   <- SNI" if m == SNI else ""))

    best = means[POST_HOC].idxmax()
    a_sni, a_best = means[SNI], means[best]
    delta = a_sni - a_best
    print(f"\nbest post-hoc comparator: {best} ({a_best:.4f})")
    print(f"Δ = A_SNI - A_best = {a_sni:.4f} - {a_best:.4f} = {delta:+.4f}"
          f"   (GO margin {GO_MARGIN:+.2f})")

    paired = piv[[SNI, best]].dropna()
    if len(paired) >= 3:
        from scipy.stats import wilcoxon
        w = wilcoxon(paired[SNI], paired[best])
        p = float(w.pvalue)
        print(f"paired Wilcoxon over {len(paired)} cells: p = {p:.5f} "
              f"({'significant' if p < ALPHA else 'not significant'} at "
              f"α={ALPHA})")
    else:
        p = float("nan")
        print(f"paired Wilcoxon: too few cells ({len(paired)})")

    # Secondary metrics, reported whether or not they agree (rule's obligation).
    print("\nsecondary metrics (mean), reported regardless of agreement:")
    sec = df.pivot_table(index="method", values=["auprc", "prec_at_k", "shd"])
    print(sec.to_string())
    disagree = []
    for col, better_is_high in (("auprc", True), ("prec_at_k", True),
                                ("shd", False)):
        s = sec[col]
        winner = s.idxmax() if better_is_high else s.idxmin()
        if (winner == SNI) != (delta >= GO_MARGIN):
            disagree.append(f"{col} favors {winner}")
    if disagree:
        print("  NOTE: " + "; ".join(disagree) +
              " — disagreement with the AUROC-based reading is itself a finding.")

    # Axis C, both framings. The rule uses total; the split is reported because
    # the rule's own reporting obligations require it.
    cost = df.pivot_table(index="method", values=["impute_sec", "audit_sec",
                                                  "total_sec"])
    print("\naxis C — cost (mean seconds per cell):")
    print(cost.to_string())
    ratio = cost.loc[SNI, "total_sec"] / cost.loc[best, "total_sec"]
    print(f"  SNI total / {best} total = {ratio:.2f}x "
          f"(CONDITIONAL GO needs <= {COST_FRACTION:.2f}x)")
    print(f"  audit-only: SNI {cost.loc[SNI, 'audit_sec']:.2f}s vs "
          f"{best} {cost.loc[best, 'audit_sec']:.2f}s "
          f"— D is free once the imputation exists, which the total-cost "
          f"framing deliberately does not credit")

    # Axis D is structural, not measured: SNI-D exists before any predictor is
    # trained; all three comparators read out of MissForest's fitted forests.
    axis_d_favours_sni = True
    print("\naxis D — dependencies: SNI-D needs neither a trained downstream "
          "model nor ground truth; all three comparators require MissForest's "
          "fitted forests. Structural, so asserted rather than measured.")

    med, per_regime, _ = cross_seed_spearman(SNI)
    print(f"\naxis B — SNI-D cross-seed stability (within regime):")
    for r, v in sorted(per_regime.items()):
        print(f"  {r:<20} median ρ = {v:.4f}")
    print(f"  overall median ρ = {med:.4f}   "
          f"(independent NO-GO below {STABILITY_FLOOR:.2f})")

    print("\n" + "=" * 72)
    if np.isfinite(med) and med < STABILITY_FLOOR:
        print(f"VERDICT: NO-GO (independent) — SNI-D's median cross-seed ρ "
              f"{med:.4f} < {STABILITY_FLOOR}. An audit tool that does not "
              f"reproduce across seeds cannot be published as one, whatever it "
              f"scores on recovery. This overrides the axis-A reading.")
        return 1
    if delta >= GO_MARGIN or (np.isfinite(p) and p >= ALPHA):
        print(f"VERDICT: GO — positioning holds as written "
              f"(Δ={delta:+.4f}, p={p:.5f}).")
        return 0
    if ratio <= COST_FRACTION and axis_d_favours_sni:
        print(f"VERDICT: CONDITIONAL GO — Δ={delta:+.4f}, p={p:.5f}, but cost "
              f"{ratio:.2f}x and no downstream model needed. The claim narrows "
              f"to a free, model-free first-order audit signal; P3 and P5 must "
              f"be rewritten accordingly.")
        return 0
    print(f"VERDICT: NO-GO — Δ={delta:+.4f} (below {GO_MARGIN:+.2f}), "
          f"p={p:.5f} (< {ALPHA}), and the cost condition fails "
          f"({ratio:.2f}x > {COST_FRACTION:.2f}x). The positioning does not "
          f"survive; P3 and P5 need rewriting and R2-1 must be answered "
          f"differently.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
