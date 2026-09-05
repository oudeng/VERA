"""The point-by-point response to the R0 reviewers, generated -- not typed.

P6. The letter quotes numbers that also appear in the manuscript, the ESM and
the evidence package. A letter whose numbers were typed by hand is a fourth
place for them to drift, and the one place nobody re-derives before
submission. So every figure here is pulled from the artifact that produced it
and formatted at render time; the prose carries `{{...}}` references, never
digits.

    PYTHONHASHSEED=2025 python reporting/response_letter.py
    PYTHONHASHSEED=2025 python reporting/response_letter.py --check <file>
    PYTHONHASHSEED=2025 python reporting/response_letter.py --selftest

--check re-renders and compares character for character, the same shape
change_summary.py and status_table.py use. It is what makes P6 SS5's
"letter numbers <-> frozen PDF" check a command rather than a proofread.

Page numbers are deliberately absent: the manuscript is not frozen, so the
letter cites \\label names and table/figure numbers, and the page column is
filled only after the freeze.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
OUT = ROOT / "reports" / "RESPONSE_TO_REVIEWERS_R1.md"

#: Where a `{{key}}` gets its value. Each entry is (artifact, dotted path,
#: format). The artifact is resolved under code_SNI unless it starts with a
#: repository-relative directory that exists at ROOT.
T_FINAL = "results/T5_stats/t_final.json"
LAMBDA = "results/T5_stats/lambda_check.json"

VALUES = {
    # --- the fair same-host pair: the paper's central comparison ---
    "fair_T":        (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.T", "+.3f"),
    "fair_T_full":   (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.T", "+.6f"),
    "fair_lo":       (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.ci95_T.0", "+.3f"),
    "fair_hi":       (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.ci95_T.1", "+.3f"),
    "fair_p":        (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.p_exact", ".4f"),
    "fair_floor":    (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.floor", ".4f"),
    "fair_seeds":    (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.n_seeds", "d"),
    # --- D versus TAP, the comparator the reviewer's item 2 asked for ---
    "dtap_T":        (T_FINAL, "recovery.D_vs_TAP.T", "+.3f"),
    "dtap_lo":       (T_FINAL, "recovery.D_vs_TAP.ci95_T.0", "+.3f"),
    "dtap_hi":       (T_FINAL, "recovery.D_vs_TAP.ci95_T.1", "+.3f"),
    "dtap_p":        (T_FINAL, "recovery.D_vs_TAP.p_exact", ".4f"),
    "dtap_neg":      (T_FINAL, "recovery.D_vs_TAP.seeds_negative", "s"),
    # --- behavioral faithfulness, 15 seeds per table ---
    "faith_mimic_T": (T_FINAL, "faithfulness.MIMIC.T", "+.3f"),
    "faith_mimic_p": (T_FINAL, "faithfulness.MIMIC.p_exact", ".4f"),
    "faith_mimic_holm": (T_FINAL, "faithfulness.MIMIC.p_holm", ".4f"),
    "faith_mimic_n": (T_FINAL, "faithfulness.MIMIC.n_seeds", "d"),
    "faith_eicu_T":  (T_FINAL, "faithfulness.eICU.T", "+.3f"),
    "faith_eicu_p":  (T_FINAL, "faithfulness.eICU.p_exact", ".4f"),
    # --- the no-prior control: the same axes without the association prior ---
    "np_mimic_T":    (T_FINAL, "noprior_faithfulness.MIMIC.T", "+.3f"),
    "np_mimic_p":    (T_FINAL, "noprior_faithfulness.MIMIC.p_exact", ".5f"),
    "np_eicu_T":     (T_FINAL, "noprior_faithfulness.eICU.T", "+.3f"),
    # --- leakage: the permutation null the reviewer asked for ---
    "null_detected": (T_FINAL, "leakage.probe_null_by_batch.original.detected", "d"),
    "null_n":        (T_FINAL, "leakage.probe_null_by_batch.original.n", "d"),
    #: p_geq_k_exact_binomial is deliberately NOT referenced. The fourth
    #: internal review ruled no fixed-alpha binomial test admissible on these
    #: counts -- the nominal level is itself estimated from a finite
    #: calibration sample -- and the terminology registry bans the rendered
    #: string. It caught this letter's first draft.
    # --- host bands under information symmetry ---
    "band_mimic":    (T_FINAL, "scoreboard_desc.host_band_mean_symmetric.MIMIC", ".3f"),
    "band_eicu":     (T_FINAL, "scoreboard_desc.host_band_mean_symmetric.eICU", ".3f"),
    # --- the gate that cannot rise ---
    "ln2":           (LAMBDA, "ln2", ".6f"),
    "lam_max":       (LAMBDA, "withprior_lambda_quantiles.max", ".6f"),
    "lam_min":       (LAMBDA, "withprior_lambda_quantiles.min", ".6f"),
    "lam_median":    (LAMBDA, "withprior_lambda_quantiles.median", ".6f"),
    "lam_models":    (LAMBDA, "n_models_scanned", "d"),
    "lam_viol":      (LAMBDA, "n_violations", "d"),
}


def _load(rel: str) -> dict:
    for base in (CODE_ROOT, ROOT):
        p = base / rel
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(rel)


def _dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def values() -> dict:
    """Every referenced number, read from the artifact that produced it."""
    cache, out = {}, {}
    for key, (rel, path, fmt) in VALUES.items():
        if rel not in cache:
            cache[rel] = _load(rel)
        v = _dig(cache[rel], path)
        if fmt == "s":
            out[key] = str(v)
        elif fmt == "d":
            out[key] = f"{int(v):d}"
        else:
            out[key] = format(float(v), fmt)
    return out


def render() -> str:
    body = (Path(__file__).parent / "response_letter_body.md").read_text()
    vals = values()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", body)) - set(vals))
    if missing:
        raise KeyError(f"the letter references values with no source: {missing}")
    unused = sorted(set(vals) - set(re.findall(r"\{\{(\w+)\}\}", body)))
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], body)
    #: A digit that reached the prose without going through a reference is
    #: exactly what this generator exists to prevent, so the check is on the
    #: SOURCE, before substitution -- afterwards every number looks typed.
    return text, unused


def check(delivered: Path) -> dict:
    text, _ = render()
    have = delivered.read_text() if delivered.exists() else None
    ok = have == text
    return {"pass": ok, "chars": len(text),
            "detail": (f"{delivered.name}: character-identical to what this "
                       f"script renders now ({len(text)} chars)" if ok else
                       f"{delivered.name}: DIFFERS from the rendered letter")}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    v = values()
    c(len(v) == len(VALUES), f"every declared value resolves ({len(v)})")
    c(v["fair_p"] == v["fair_floor"],
      "the fair pair's p equals its own attainable floor -- the letter must "
      "not read this as a near-miss")
    c(v["fair_T"].startswith("+"), "signed effects render their sign")
    c(float(v["lam_max"]) <= float(v["ln2"]),
      "the scanned gate maximum does not exceed ln 2")
    body = (Path(__file__).parent / "response_letter_body.md")
    c(body.exists(), "the letter body is present")
    if body.exists():
        raw = body.read_text()
        #: The reviewers' own numbers are QUOTATIONS and must stay verbatim --
        #: they are what the reviewer wrote, not what our artifacts say, and
        #: regenerating them would be a quiet edit of someone else's words.
        #: Blockquoted lines are therefore exempt; everywhere else, a typed
        #: decimal is the defect this generator exists to prevent.
        ours = "\n".join(l for l in raw.splitlines()
                          if not l.lstrip().startswith(">"))
        stray = re.findall(r"(?<![\w.=/#§-])\d+\.\d+(?![\w.])", ours)
        c(not stray, f"no decimal is typed outside a quotation: {stray[:6]}")
        quoted = re.findall(r"(?<![\w.=/#-])\d+\.\d+(?![\w.])",
                            "\n".join(l for l in raw.splitlines()
                                       if l.lstrip().startswith(">")))
        c(bool(quoted),
          f"the reviewers' own figures are quoted verbatim ({len(quoted)} found)")
        text, unused = render()
        c("{{" not in text, "every reference was substituted")
        c(not unused, f"no declared value goes unused: {unused}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--values", action="store_true",
                    help="print every reference and the artifact behind it")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.values:
        for k, (rel, path, fmt) in sorted(VALUES.items()):
            print(f"{k:16} {values()[k]:>12}   <- {rel}#{path}")
        return 0
    if a.check:
        r = check(Path(a.check))
        print(("[OK] " if r["pass"] else "[RED] ") + r["detail"])
        return 0 if r["pass"] else 1
    text, _ = render()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"[ok] wrote {OUT} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
